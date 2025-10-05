# app.py
from __future__ import annotations

import io
import json
import sys
import traceback
from pathlib import Path
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin

# -----------------------------------------------------------------------------
# FastAPI + CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="Nalanda Nexus AI - Exoplanet API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.nalandanexusai.work",      # your custom domain
        "https://nalandanexusai.work",
        "https://MohapatraSaloni.github.io", # your GitHub Pages root
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Backward-compatible PercentileClipper
# -----------------------------------------------------------------------------
class PercentileClipper(BaseEstimator, TransformerMixin):
    """
    Clips numeric columns to [lower, upper] percentiles per feature.

    Backward-compatible with older pickles that may:
      - miss 'lower'/'upper' attrs
      - store bounds as 'lower_bounds'/'upper_bounds' (no trailing underscore)
      - lack fitted bounds entirely (we recompute from X as last resort)
    """

    def __init__(self, lower: float = 1.0, upper: float = 99.0):
        # Defaults for fresh instances
        self.lower = float(lower)
        self.upper = float(upper)
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None

    # When loading from pickle, ensure required attributes exist
    def __setstate__(self, state):
        # Populate this instance with whatever the pickle had
        self.__dict__.update(state)

        # Make sure lower/upper exist (use defaults if they don't)
        if not hasattr(self, "lower") or self.lower is None:
            self.lower = 1.0
        if not hasattr(self, "upper") or self.upper is None:
            self.upper = 99.0

        # Normalize bounds attribute names for compatibility
        if not hasattr(self, "lower_bounds_") and hasattr(self, "lower_bounds"):
            self.lower_bounds_ = getattr(self, "lower_bounds")
        if not hasattr(self, "upper_bounds_") and hasattr(self, "upper_bounds"):
            self.upper_bounds_ = getattr(self, "upper_bounds")

    def _as_ndarray(self, X):
        if hasattr(X, "values"):  # pandas DataFrame
            return X.values.astype(float)
        return np.asarray(X, dtype=float)

    def fit(self, X, y=None):
        X_arr = self._as_ndarray(X)
        # Ensure lower/upper exist even if user overwrote them
        lower = float(getattr(self, "lower", 1.0))
        upper = float(getattr(self, "upper", 99.0))
        self.lower_bounds_ = np.nanpercentile(X_arr, lower, axis=0)
        self.upper_bounds_ = np.nanpercentile(X_arr, upper, axis=0)
        # Also set old names for safety
        setattr(self, "lower_bounds", self.lower_bounds_)
        setattr(self, "upper_bounds", self.upper_bounds_)
        return self

    def _get_bounds(self, X):
        # Prefer new names
        lb = getattr(self, "lower_bounds_", None)
        ub = getattr(self, "upper_bounds_", None)
        # Fallback to old names
        if lb is None:
            lb = getattr(self, "lower_bounds", None)
        if ub is None:
            ub = getattr(self, "upper_bounds", None)

        # Ensure lower/upper exist
        lower = float(getattr(self, "lower", 1.0))
        upper = float(getattr(self, "upper", 99.0))

        # Last resort: compute from X to avoid crashing
        if lb is None or ub is None:
            X_arr = self._as_ndarray(X)
            lb = np.nanpercentile(X_arr, lower, axis=0)
            ub = np.nanpercentile(X_arr, upper, axis=0)
            self.lower_bounds_ = lb
            self.upper_bounds_ = ub
            setattr(self, "lower_bounds", lb)
            setattr(self, "upper_bounds", ub)

        return lb, ub

    def transform(self, X):
        X_arr = self._as_ndarray(X)
        lb, ub = self._get_bounds(X_arr)
        lb = np.broadcast_to(lb, X_arr.shape)
        ub = np.broadcast_to(ub, X_arr.shape)
        return np.clip(X_arr, lb, ub)

# Make pickled references like "__main__.PercentileClipper" resolvable
sys.modules.setdefault("__main__", sys.modules[__name__])
setattr(sys.modules["__main__"], "PercentileClipper", PercentileClipper)

# -----------------------------------------------------------------------------
# Model + threshold loading
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "lgbm_pipeline.pkl"
THR_PATH = BASE_DIR / "models" / "threshold.json"

model = None
DEFAULT_THR = 0.5  # fallback if threshold.json missing

def _safe_load_model():
    global model, DEFAULT_THR
    if not MODEL_PATH.exists():
        print(f"⚠️  Model not found at: {MODEL_PATH}")
        return
    try:
        print(f"⏳ Loading model from: {MODEL_PATH}")
        mdl = joblib.load(MODEL_PATH)
        print("✅ Model loaded.")
        model = mdl
        if THR_PATH.exists():
            with open(THR_PATH, "r") as f:
                data = json.load(f)
                DEFAULT_THR = float(data.get("threshold", DEFAULT_THR))
                print(f"✅ Threshold loaded: {DEFAULT_THR:.4f}")
        else:
            print(f"⚠️  Threshold file not found at: {THR_PATH}. Using DEFAULT_THR={DEFAULT_THR:.2f}")
    except Exception as e:
        print("🚨 Failed to load model:", repr(e))
        traceback.print_exc()

_safe_load_model()

# -----------------------------------------------------------------------------
# Introspect pipeline expected columns (if a ColumnTransformer exists)
# -----------------------------------------------------------------------------
def expected_columns_from_pipeline(m) -> list | None:
    try:
        steps = getattr(m, "named_steps", {})
        ct = steps.get("preprocess") or steps.get("preprocessor") or steps.get("prep") or None
        if ct is None:
            return None
        expected = []
        for _, _, cols in getattr(ct, "transformers_", []):
            if cols in (None, "drop", "passthrough"):
                continue
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                expected.extend(list(cols))
        # de-duplicate preserving order
        return list(dict.fromkeys(expected))
    except Exception:
        return None

EXPECTED_COLS = expected_columns_from_pipeline(model) if model is not None else None
print("➡️  Pipeline expected columns:", EXPECTED_COLS)

# -----------------------------------------------------------------------------
# Schemas + feature building
# -----------------------------------------------------------------------------
FEATURE_ORDER: List[str] = [
    "period_days",
    "duration_hours",
    "depth_ppm",
    "snr",
    "teff",
    "mag",
]

ENGINEERED_COLS = ["depth_per_hour", "log_depth", "log_period", "log_snr"]

class ExoplanetInput(BaseModel):
    period_days: float
    duration_hours: float
    depth_ppm: float
    snr: float
    teff: float
    mag: float
    threshold: Optional[float] = None

def _ensure_model_loaded():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Place 'lgbm_pipeline.pkl' in ./models and restart.",
        )

def _df_from_input(inp: ExoplanetInput) -> pd.DataFrame:
    row = {k: getattr(inp, k) for k in FEATURE_ORDER}
    return pd.DataFrame([row], columns=FEATURE_ORDER)

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate training-time engineered features.
    If you trained with np.log or np.log10, change these lines to match exactly.
    """
    out = df.copy()
    eps = 1e-9
    out["depth_per_hour"] = out["depth_ppm"] / (out["duration_hours"] + eps)
    out["log_depth"]  = np.log1p(out["depth_ppm"].clip(lower=0))
    out["log_period"] = np.log1p(out["period_days"].clip(lower=0))
    out["log_snr"]    = np.log1p(out["snr"].clip(lower=0))
    return out

def _order_like_pipeline(X: pd.DataFrame) -> pd.DataFrame:
    if EXPECTED_COLS:
        missing = set(EXPECTED_COLS) - set(X.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"Engineered columns missing: {missing}")
        return X.loc[:, EXPECTED_COLS]
    return X

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/ping")
def ping():
    return {"ok": True}

@app.post("/api/predict-single")
def predict_single(input: ExoplanetInput, request: Request):
    """
    JSON body:
    {
      "period_days": 3.0, "duration_hours": 2.0, "depth_ppm": 600,
      "snr": 12, "teff": 5800, "mag": 11.5, "threshold": null
    }
    """
    _ensure_model_loaded()
    try:
        X_base = _df_from_input(input)
        X_full = add_engineered_features(X_base)
        X_full = _order_like_pipeline(X_full)

        proba = float(model.predict_proba(X_full)[:, 1][0])
        thr = float(input.threshold) if input.threshold is not None else float(DEFAULT_THR)
        pred = int(proba >= thr)

        return {"ok": True, "score": proba, "threshold": thr, "predicted_class": pred}

    except HTTPException:
        raise
    except Exception as e:
        print("🚨 Error during /api/predict-single:", e)
        print("   ✳️ Columns raw:", list(X_base.columns))
        print("   ✳️ Columns engineered:", list(X_full.columns) if 'X_full' in locals() else None)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict-batch")
async def predict_batch(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
):
    """
    form-data:
      file: CSV with columns (period_days,duration_hours,depth_ppm,snr,teff,mag)
      threshold: optional float; uses default if omitted
    """
    _ensure_model_loaded()
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))

        base_required = FEATURE_ORDER
        miss_base = [c for c in base_required if c not in df.columns]
        if miss_base:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {miss_base}")

        X_base = df.loc[:, base_required]
        X_full = add_engineered_features(X_base)
        X_full = _order_like_pipeline(X_full)

        y_score = model.predict_proba(X_full)[:, 1]
        thr = float(threshold) if threshold is not None else float(DEFAULT_THR)
        y_pred = (y_score >= thr).astype(int)

        out = df.copy()
        out["p_planet"] = y_score
        out["label_pred"] = y_pred

        return {"ok": True, "threshold_used": thr, "csv": out.to_csv(index=False)}

    except HTTPException:
        raise
    except Exception as e:
        print("🚨 Error during /api/predict-batch:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


