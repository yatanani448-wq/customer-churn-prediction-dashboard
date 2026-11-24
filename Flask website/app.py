"""Flask Customer Churn Prediction and Analytics Dashboard.

This application reads the existing model and dataset from its parent project;
it never retrains, replaces, or writes to those ML assets.
"""
from functools import lru_cache
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from flask import Flask, jsonify, render_template, request
from plotly.utils import PlotlyJSONEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
DATASET_PATH = PROJECT_ROOT / "cleaned_customer_churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "customer_churn_model.pkl"
TARGET = "Churn"
SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
DERIVED_FEATURES = {"TenureGroup", "ServiceCount"}

app = Flask(__name__)


def add_notebook_features(frame):
    """Apply the feature engineering from smlfullproject.ipynb exactly."""
    enriched = frame.copy()
    if "TenureGroup" not in enriched.columns:
        enriched["TenureGroup"] = pd.cut(
            enriched["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"],
        )
    if "ServiceCount" not in enriched.columns:
        enriched["ServiceCount"] = sum(
            (enriched[column] == "Yes").astype(int) for column in SERVICE_COLUMNS
        )
    return enriched


@lru_cache(maxsize=1)
def assets():
    if not DATASET_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError("The existing cleaned dataset or trained model is missing from the parent project.")
    data = add_notebook_features(pd.read_csv(DATASET_PATH))
    model = joblib.load(MODEL_PATH)
    features = list(getattr(model, "feature_names_in_", []))
    if not features:
        features = list(model.named_steps["preprocessor"].feature_names_in_)
    missing = [column for column in features if column not in data.columns]
    if missing:
        raise ValueError("Dataset does not contain model feature(s): " + ", ".join(missing))
    return data, model, features


def chart_json(figure):
    figure.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=55, b=30), font=dict(family="Inter, Arial, sans-serif"))
    return json.dumps(figure, cls=PlotlyJSONEncoder)


def summary(data):
    return {
        "customers": len(data),
        "churned": int(data[TARGET].sum()),
        "churn_rate": float(data[TARGET].mean() * 100),
        "avg_monthly": float(data["MonthlyCharges"].mean()) if "MonthlyCharges" in data else 0,
    }


def model_probability(model, frame):
    if not hasattr(model, "predict_proba"):
        return None
    classes = list(model.classes_)
    positive_index = classes.index(1) if 1 in classes else len(classes) - 1
    return float(model.predict_proba(frame)[:, positive_index][0])


LABELS = {
    "gender": "Gender", "SeniorCitizen": "Senior Citizen", "Partner": "Partner", "Dependents": "Dependents", "tenure": "Tenure (Months)",
    "PhoneService": "Phone Service", "MultipleLines": "Multiple Lines", "InternetService": "Internet Service",
    "OnlineSecurity": "Online Security", "OnlineBackup": "Online Backup", "DeviceProtection": "Device Protection", "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming Movies", "Contract": "Contract", "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method", "MonthlyCharges": "Monthly Charges", "TotalCharges": "Total Charges",
}
HELP_TEXT = {
    "SeniorCitizen": "Yes means age 65 or older.",
    "tenure": "Months the customer has been with the company.",
    "InternetService": "DSL is standard broadband. Fiber optic is high-speed internet. No means no internet service.",
    "OnlineSecurity": "No internet service means the customer has no internet access.",
    "OnlineBackup": "No internet service means the customer has no internet access.",
    "DeviceProtection": "No internet service means the customer has no internet access.",
    "TechSupport": "No internet service means the customer has no internet access.",
    "StreamingTV": "No internet service means the customer has no internet access.",
    "StreamingMovies": "No internet service means the customer has no internet access.",
    "Contract": "Month-to-month has no long-term commitment.",
    "TotalCharges": "Total amount billed to the customer so far.",
}
FORM_SECTIONS = [
    ("Customer Information", "Basic customer and household details.", ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure"]),
    ("Phone and Internet Services", "The customer's connectivity services.", ["PhoneService", "MultipleLines", "InternetService"]),
    ("Additional Services", "Services included with the internet plan.", ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]),
    ("Billing Information", "Contract, billing preferences, and charges.", ["Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]),
]


def form_sections(data, features):
    raw = [feature for feature in features if feature not in DERIVED_FEATURES]
    fields = {}
    for feature in raw:
        series = data[feature]
        numeric = pd.api.types.is_numeric_dtype(series)
        fields[feature] = {
            "name": feature, "label": LABELS.get(feature, feature), "hint": HELP_TEXT.get(feature),
            "numeric": numeric, "integer": pd.api.types.is_integer_dtype(series),
            "default": int(series.median()) if numeric and pd.api.types.is_integer_dtype(series) else (round(float(series.median()), 2) if numeric else str(series.mode().iat[0])),
            "options": [] if numeric else sorted(series.dropna().astype(str).unique().tolist()),
        }
    result = []
    assigned = set()
    for heading, description, names in FORM_SECTIONS:
        available = [fields[name] for name in names if name in fields]
        assigned.update(field["name"] for field in available)
        if available:
            result.append((heading, description, available))
    extra = [fields[name] for name in raw if name not in assigned]
    if extra:
        result.append(("Additional Model Inputs", "Further information required by the saved model.", extra))
    return result


@app.context_processor
def inject_navigation():
    return {"nav_items": [("Home", "home"), ("Dashboard", "dashboard"), ("Customer Analytics", "analytics"), ("Predict Churn", "predict"), ("Model Performance", "performance"), ("About", "about")]}


@app.route("/")
def home():
    data, _, _ = assets()
    return render_template("home.html", summary=summary(data), page="home")


@app.route("/dashboard")
def dashboard():
    data, _, _ = assets()
    contract = data.groupby("Contract", as_index=False)[TARGET].mean() if "Contract" in data else pd.DataFrame()
    contract_fig = px.bar(contract, x="Contract", y=TARGET, text_auto=".1%", color_discrete_sequence=["#2563eb"], title="Churn Rate by Contract")
    contract_fig.update_yaxes(tickformat=".0%", title="Churn rate")
    distribution = data[TARGET].map({0: "Stayed", 1: "Churned"}).value_counts().rename_axis("Outcome").reset_index(name="Customers")
    churn_fig = px.pie(distribution, names="Outcome", values="Customers", hole=.65, color="Outcome", color_discrete_map={"Stayed": "#2563eb", "Churned": "#f97316"}, title="Customer Churn Distribution")
    return render_template("dashboard.html", summary=summary(data), page="dashboard", contract_chart=chart_json(contract_fig), churn_chart=chart_json(churn_fig))


@app.route("/analytics")
def analytics():
    data, _, _ = assets()
    available = [column for column in ["Contract", "InternetService", "PaymentMethod", "tenure", "MonthlyCharges", "TotalCharges"] if column in data]
    return render_template("analytics.html", page="analytics", dimensions=available)


@app.route("/api/analytics")
def analytics_api():
    data, _, _ = assets()
    dimension = request.args.get("dimension", "Contract")
    if dimension not in data.columns:
        return jsonify({"error": "Unknown analytics dimension."}), 400
    if pd.api.types.is_numeric_dtype(data[dimension]):
        figure = px.histogram(data, x=dimension, color=TARGET, nbins=30, barmode="overlay", color_discrete_map={0: "#2563eb", 1: "#f97316"}, title=f"Churn Pattern by {dimension}")
    else:
        grouped = data.groupby(dimension, as_index=False)[TARGET].mean().sort_values(TARGET, ascending=False)
        figure = px.bar(grouped, x=dimension, y=TARGET, text_auto=".1%", color_discrete_sequence=["#2563eb"], title=f"Churn Rate by {dimension}")
        figure.update_yaxes(tickformat=".0%", title="Churn rate")
    return jsonify(json.loads(chart_json(figure)))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    data, model, features = assets()
    result = None
    error = None
    if request.method == "POST":
        try:
            values = {}
            for feature in [item for item in features if item not in DERIVED_FEATURES]:
                if feature == "SeniorCitizen":
                    values[feature] = 1 if request.form.get(feature) == "Yes" else 0
                elif pd.api.types.is_numeric_dtype(data[feature]):
                    values[feature] = int(request.form[feature]) if pd.api.types.is_integer_dtype(data[feature]) else float(request.form[feature])
                else:
                    values[feature] = request.form[feature]
            record = add_notebook_features(pd.DataFrame([values]))[features]
            prediction = int(model.predict(record)[0])
            result = {"label": "High churn risk" if prediction else "Low churn risk", "churned": bool(prediction), "probability": model_probability(model, record)}
        except (KeyError, TypeError, ValueError) as exc:
            error = f"Please check the form values: {exc}"
    return render_template("predict.html", page="predict", sections=form_sections(data, features), result=result, error=error)


@app.route("/performance")
def performance():
    data, model, features = assets()
    x, y = data[features], data[TARGET].astype(int)
    predicted = model.predict(x).astype(int)
    probabilities = model.predict_proba(x)[:, list(model.classes_).index(1)] if hasattr(model, "predict_proba") else None
    metrics = {"Accuracy": accuracy_score(y, predicted), "Precision": precision_score(y, predicted, zero_division=0), "Recall": recall_score(y, predicted, zero_division=0), "F1 Score": f1_score(y, predicted, zero_division=0)}
    if probabilities is not None and y.nunique() == 2:
        metrics["ROC-AUC"] = roc_auc_score(y, probabilities)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    figure = px.imshow(matrix, text_auto=True, x=["Predicted stayed", "Predicted churned"], y=["Actual stayed", "Actual churned"], color_continuous_scale="Blues", title="Confusion Matrix")
    figure.update_layout(coloraxis_showscale=False)
    return render_template("performance.html", page="performance", metrics=metrics, matrix_chart=chart_json(figure), records=len(data))


@app.route("/about")
def about():
    data, model, features = assets()
    return render_template("about.html", page="about", records=len(data), features=len(features), model_type=type(model).__name__)


if __name__ == "__main__":
    app.run(debug=True)
