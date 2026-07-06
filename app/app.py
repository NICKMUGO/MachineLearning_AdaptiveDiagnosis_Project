"""
Streamlit demo — Adaptive Symptom-to-Treatment Recommender.
Run from repo root: streamlit run app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import streamlit as st

from src.pipeline import get_top_k_conditions
from src.feedback import log_feedback, retrain_on_feedback

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / 'data' / 'processed'

st.set_page_config(
    page_title='Adaptive Symptom-to-Treatment Recommender',
    page_icon='🩺',
    layout='wide',
)

st.title('Adaptive Symptom-to-Treatment Recommender')
st.error(
    '**Disclaimer:** This is an academic / educational demonstration trained on limited '
    'secondary data. It is **not a medical device** and must not replace professional '
    'medical advice. Always consult a qualified healthcare provider.'
)


@st.cache_resource
def get_feature_cols():
    p = PROC / 'feature_cols.pkl'
    return joblib.load(p) if p.exists() else None


feature_cols = get_feature_cols()

if feature_cols is None:
    st.error('Model not trained yet. Run notebooks 01–04 first to generate the model artifacts.')
    st.stop()

symptom_options = sorted(c.replace('_', ' ').strip().title() for c in feature_cols)
raw_map = {c.replace('_', ' ').strip().title(): c for c in feature_cols}
st.header('Step 1 — Select Symptoms')
selected_display = st.multiselect(
    'Choose all symptoms that apply:',
    options=symptom_options,
    help='Select one or more symptoms from the list.',
)
k = st.slider('Number of conditions to consider (top-k)', min_value=1, max_value=5, value=3)

if 'results' not in st.session_state:
    st.session_state.results = None

if st.button('Get Recommendations', type='primary', disabled=not selected_display):
    selected_raw = [raw_map[d] for d in selected_display]
    with st.spinner('Analysing symptoms…'):
        try:
            top_k, treatments = get_top_k_conditions(selected_raw, k=k)
            st.session_state.results = (selected_raw, top_k, treatments)
        except FileNotFoundError as exc:
            st.error(f'Model artifact missing: {exc}. Run notebooks 01–04 first.')

if st.session_state.results:
    selected_raw, top_k, treatments = st.session_state.results

    st.divider()
    st.header('Step 2 — Differential Assessment')
    st.caption(
        'The system never commits to a single diagnosis. '
        'These are the most consistent possibilities given your symptoms.'
    )
    cols = st.columns(len(top_k))
    for col, (disease, prob) in zip(cols, top_k):
        col.metric(disease.title(), f'{prob * 100:.1f}%')
        col.progress(prob)

    st.header('Step 3 — Recommended Management Steps')
    for disease, _ in top_k:
        steps = treatments.get(disease, [])
        with st.expander(f'{disease.title()}', expanded=True):
            if steps:
                for i, step in enumerate(steps, 1):
                    st.write(f'{i}. {step.capitalize()}')
            else:
                st.write('No specific precautions on record for this condition.')

    st.divider()
    st.header('Step 4 — Outcome Feedback')
    st.write('Your feedback is used to improve recommendations over time.')

    outcome = st.radio(
        'How did the recommendations work out?',
        ['improved', 'no change', 'worse', 'symptoms changed'],
        horizontal=True,
    )

    new_symptoms_raw = []
    if outcome == 'symptoms changed':
        new_display = st.multiselect(
            'Select your updated symptoms:',
            options=symptom_options,
            key='new_syms',
        )
        new_symptoms_raw = [raw_map[d] for d in new_display]

    if st.button('Submit Feedback'):
        log_feedback(selected_raw, top_k, treatments, outcome, new_symptoms_raw)
        _, msg = retrain_on_feedback()
        st.success(f'Feedback recorded. {msg}')
