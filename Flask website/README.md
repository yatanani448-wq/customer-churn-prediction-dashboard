# Flask Customer Churn Dashboard

This folder contains the Flask migration only. It reads the existing model and dataset from the parent project; it does not modify them.

## Run locally (Windows)

Open a terminal in this `Flask website` folder and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

The saved model was created with scikit-learn 1.6.1, which is pinned in `requirements.txt`. Do not upgrade that package independently, because newer scikit-learn releases may not load the saved pipeline.
