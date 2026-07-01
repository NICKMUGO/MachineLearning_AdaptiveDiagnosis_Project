# Adaptive Symptom-to-Treatment Recommender

> **Academic / Educational Project** — This system is a coursework demonstration trained on limited secondary data. It is **not a medical device** and must not be used as a substitute for professional medical advice.

A machine-learning system that takes a patient's symptoms, maintains a probabilistic differential across several possible conditions (rather than committing to one), recommends ranked treatment/management steps, and improves over time as patients report outcomes.

---

## The Core Idea

> *Symptoms can be misleading and may point to an undiscovered illness — so never lock in a single diagnosis.*

Instead of outputting a single disease label, the system says:

> "Your symptoms are most consistent with **A (52%)**, **B (31%)**, **C (17%)**. Recommended steps that help across these: …"

```
Symptoms ──► Probabilistic differential ──► Ranked treatment / ──► Patient outcome
             (top-k conditions,               management recs        feedback
              never just one)                                     (recovered? changed?)
                    │
                    ◄──────────────────────────────────────────┘
                              model + ranking update over time
```

---

## System Architecture

| Layer | What it does | Method |
|---|---|---|
| **1. Diagnosis (differential)** | Symptoms → probabilities over conditions, keep top-k | Ensemble: Random Forest + XGBoost + Logistic Regression (soft-voting / stacking) |
| **2. Treatment mapping** | Conditions → ranked treatment/management/precautions | Rule-based lookup from precaution/drug-review tables, weighted by condition probability |
| **3. Dynamic feedback** | New cases + outcomes update the system | Incremental learning (`partial_fit` / `river`) and/or periodic retraining on a growing case store |
| **Feedback store** | Logs every case + outcome | SQLite or appended CSV: `{symptoms, recommended, outcome, new_symptoms, timestamp}` |

---

## Datasets

| Dataset | Source | Use |
|---|---|---|
| Disease Prediction Using ML (`Training.csv`, `Testing.csv`) | Kaggle — kaushil268 | Layer 1: 132 symptoms → 42 diseases |
| Disease Symptom Prediction (`symptom_precaution.csv`, etc.) | Kaggle — itachi9604 | Layer 2: disease → recommended actions |
| Drug Review Dataset (Drugs.com) | UCI / Kaggle | Treatment ranking by patient-reported effectiveness |
| DDXPlus *(stretch)* | Kaggle / figshare | ~1.3M synthetic cases with ground-truth differential — justifies top-k design |
| Symptom2Disease *(stretch)* | Kaggle | NLP free-text symptom input demo |

**Recommended combo for 7 days:** kaushil268 (modeling) + itachi9604 precautions (treatment) + Drugs.com reviews (treatment ranking).

---

## Tech Stack

| Purpose | Libraries |
|---|---|
| Data / EDA | `pandas`, `numpy` |
| Modeling | `scikit-learn`, `xgboost` |
| Online / incremental learning | `scikit-learn` (`partial_fit`) or `river` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| App | `streamlit` |
| Storage | `sqlite3` / CSV |

---

## Repo Structure

```
project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            # downloaded datasets (gitignored)
│   └── processed/
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_ensemble_treatment.ipynb
│   └── 05_dynamic_learning.ipynb
├── src/
│   ├── pipeline.py     # symptoms -> top-k -> treatments
│   └── feedback.py     # feedback store + retrain
└── app/
    └── app.py          # streamlit demo
```

---

## Quick Start

```bash
# clone and enter the repo
git clone <repo-url>
cd MachineLearning_AdaptiveDiagnosis_Project

# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# run the streamlit app
streamlit run app/app.py
```

---

## Dynamic / "Learns as You Feed It" Design

Two mechanisms work together:

**A. Feedback-loop retraining (primary)**
Every recommendation triggers a user outcome report (`improved / no change / worse / symptoms changed`). The case is appended to the feedback store and the model periodically retrains (or `partial_fit`s) on the accumulated data. Treatments that correlate with "improved" outcomes float to the top for similar symptom profiles.

**B. True online learning (advanced)**
Using `river` or `sklearn`'s `partial_fit`, the model updates one record at a time as data streams in — no full retrain required. The patient journey is modelled as a sequence: visit 1 (symptoms + treatment + outcome) → visit 2 (new symptoms) → …

The **killer demo visual** is an incremental learning curve — accuracy/F1 plotted as more batches are fed — proving the "it learns as you feed it" claim in one chart.

---

## Evaluation

- **Top-k accuracy** (top-3 matters more than top-1 for a differential system)
- Macro-F1, per-class precision / recall
- Single model vs. ensemble comparison
- Accuracy before vs. after feeding new batches (dynamic learning proof)
- Stratified splits + `class_weight` for class imbalance

---

## Visualizations

**EDA**
- Disease/class distribution (bar chart) — check for imbalance
- Top-20 most common symptoms
- Symptom co-occurrence / correlation heatmap
- Symptom-severity distribution

**Results**
- Confusion matrix (or top-k accuracy)
- Per-class precision / recall / F1
- Feature importance (which symptoms drive predictions)
- ROC / Precision-Recall curves
- Incremental learning curve (dynamic-learning proof)

---

## 7-Day Sprint Plan

| Day | Focus | Deliverable |
|---|---|---|
| 1 | Setup & data | Repo live, data loaded, `01_data_loading.ipynb` runs |
| 2 | EDA & cleaning | `02_eda.ipynb` with charts, clean processed dataset |
| 3 | Baseline model | `03_baseline.ipynb`, saved models, metrics table |
| 4 | Ensemble + treatment layer | Working symptoms → top-k → ranked treatments pipeline |
| 5 | Dynamic learning + feedback loop | `05_dynamic_learning.ipynb`, feedback store, learning-curve viz |
| 6 | App + integration + results viz | Running app (local or deployed), final charts, frozen main branch |
| 7 | Presentation & dry run | Finished deck + rehearsed presentation, polished README |

---

## Honest Risks & Caveats

- **Data is not clinical reality.** Public symptom datasets are small, often synthetic, and don't reflect real diagnostic complexity.
- **Treatments are condition-indexed.** The system is "differential-aware treatment suggestion," not a magic symptom-to-cure engine.
- **The feedback loop can reinforce bias.** If early users over-report one outcome, rankings skew. Real-world deployment would need safeguards.
- **Full RL for treatment is research-grade.** This project implements incremental learning + outcome-weighted ranking — an honest, achievable version.
- **Not a medical device.** Disclaimer is built into the app and the presentation.

---

## License

This project is for academic and educational use only.
