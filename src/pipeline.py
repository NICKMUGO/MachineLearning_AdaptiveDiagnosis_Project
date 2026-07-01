"""
symptoms -> top-k conditions -> ranked treatments
Requires data/processed/ artifacts produced by notebooks 02-04.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / 'data' / 'processed'


def _norm(name: str) -> str:
    return name.strip().lower().replace(' ', '_')


def load_artifacts():
    model        = joblib.load(PROC / 'ensemble_model.pkl')
    feature_cols = joblib.load(PROC / 'feature_cols.pkl')
    le           = joblib.load(PROC / 'label_encoder.pkl')
    prec_df      = pd.read_csv(ROOT / 'data' / 'raw' / 'itachi9604' / 'symptom_precaution.csv')
    prec_df.columns = [c.strip() for c in prec_df.columns]
    prec_df['Disease'] = prec_df['Disease'].str.strip().str.lower()
    return model, feature_cols, le, prec_df


def symptoms_to_vector(symptoms: list, feature_cols: list) -> np.ndarray:
    vec     = np.zeros(len(feature_cols))
    sym_set = {_norm(s) for s in symptoms}
    for i, col in enumerate(feature_cols):
        if _norm(col) in sym_set:
            vec[i] = 1
    return vec.reshape(1, -1)


def get_top_k_conditions(symptoms: list, k: int = 3):
    """Return (top_k, treatments) where top_k is [(disease, prob), ...]."""
    model, feature_cols, le, prec_df = load_artifacts()
    vec    = symptoms_to_vector(symptoms, feature_cols)
    probs  = model.predict_proba(vec)[0]
    top_idx = np.argsort(probs)[::-1][:k]
    top_k   = [(le.inverse_transform([i])[0], round(float(probs[i]), 4)) for i in top_idx]
    treatments = get_treatments(top_k, prec_df)
    return top_k, treatments


def get_treatments(top_k_conditions: list, prec_df: pd.DataFrame) -> dict:
    prec_cols = [c for c in prec_df.columns if 'precaution' in c.lower()]
    results   = {}
    for disease, _ in top_k_conditions:
        match = prec_df[prec_df['Disease'] == disease.strip().lower()]
        if not match.empty:
            steps = [
                str(v).strip()
                for col in prec_cols
                for v in match[col].values
                if pd.notna(v) and str(v).strip() not in ('', 'nan')
            ]
            results[disease] = list(dict.fromkeys(steps))
        else:
            results[disease] = []
    return results
