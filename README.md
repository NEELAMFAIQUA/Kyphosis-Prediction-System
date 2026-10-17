# Kyphosis Classifier: Synthetic Teaching Demo

This original interactive lesson demonstrates a binary-classification workflow
using **generated synthetic data only**. It does not train on patient records,
accept clinical measurements, or predict an individual's health outcome.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The synthetic accuracy shown is only a coding exercise and has no clinical
meaning. This is not a diagnostic tool; do not use it for healthcare decisions.
Run tests with `python -m unittest discover -s tests -v`.
