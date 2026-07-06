"""
Feedback store (SQLite) + incremental partial_fit update.
"""
import sqlite3
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / 'data' / 'feedback.db'
PROC    = ROOT / 'data' / 'processed'


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp              TEXT NOT NULL,
            symptoms               TEXT NOT NULL,
            top_conditions         TEXT NOT NULL,
            recommended_treatments TEXT NOT NULL,
            outcome                TEXT NOT NULL,
            new_symptoms           TEXT
        )
    """)
    con.commit()
    con.close()


def log_feedback(symptoms, top_conditions, recommended_treatments, outcome, new_symptoms=None):
    init_db()
    con = sqlite3.connect(DB_PATH)
    con.execute(
        'INSERT INTO feedback VALUES (NULL,?,?,?,?,?,?)',
        (
            datetime.utcnow().isoformat(),
            json.dumps(symptoms),
            json.dumps(top_conditions),
            json.dumps(recommended_treatments),
            outcome,
            json.dumps(new_symptoms or []),
        ),
    )
    con.commit()
    con.close()


def load_feedback() -> pd.DataFrame:
    init_db()
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql('SELECT * FROM feedback', con)
    con.close()
    return df


def retrain_on_feedback():
    """Partial-fit an SGDClassifier on all 'improved' feedback cases."""
    from sklearn.linear_model import SGDClassifier

    df = load_feedback()
    if df.empty:
        return None, 'No feedback data yet.'

    improved = df[df['outcome'] == 'improved']
    if improved.empty:
        return None, "No 'improved' cases logged yet."

    feat_path = PROC / 'feature_cols.pkl'
    enc_path  = PROC / 'label_encoder.pkl'
    if not feat_path.exists() or not enc_path.exists():
        return None, 'Model artifacts missing — run notebooks 02-04 first.'

    feature_cols = joblib.load(feat_path)
    le           = joblib.load(enc_path)
    n_classes    = len(le.classes_)

    online_path = PROC / 'online_model.pkl'
    online_model = (
        joblib.load(online_path) if online_path.exists()
        else SGDClassifier(loss='log_loss', random_state=42)
    )
    # If the label encoder or feature set has since changed shape (e.g.
    # notebooks 02-04 rerun on new data), the previously fitted
    # online_model's classes_/n_features_in_ no longer match what
    # partial_fit would reject as a change from the first call, so start
    # a fresh model instead of reusing the stale one.
    stale = (
        (hasattr(online_model, 'classes_') and len(online_model.classes_) != n_classes)
        or (hasattr(online_model, 'n_features_in_') and online_model.n_features_in_ != len(feature_cols))
    )
    if stale:
        online_model = SGDClassifier(loss='log_loss', random_state=42)

    X_rows, y_rows = [], []
    for _, row in improved.iterrows():
        symptoms = json.loads(row['symptoms'])
        vec      = np.zeros(len(feature_cols))
        sym_set  = {s.strip().lower().replace(' ', '_') for s in symptoms}
        for i, col in enumerate(feature_cols):
            if col.strip().lower().replace(' ', '_') in sym_set:
                vec[i] = 1

        top_conditions = json.loads(row['top_conditions'])
        if top_conditions:
            top_label = top_conditions[0][0]
            try:
                y_enc = le.transform([top_label])[0]
                X_rows.append(vec)
                y_rows.append(y_enc)
            except ValueError:
                pass

    if not X_rows:
        return None, 'No valid rows to train on.'

    X = np.array(X_rows)
    y = np.array(y_rows)
    online_model.partial_fit(X, y, classes=np.arange(n_classes))
    joblib.dump(online_model, online_path)
    return online_model, f'Online model updated on {len(X_rows)} improved cases.'
