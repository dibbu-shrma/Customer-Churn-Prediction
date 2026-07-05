import pickle
import json
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

label_encoded_columns = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "PaperlessBilling"
]

label_maps = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
    "MultipleLines": {"No": 0, "No phone service": 1, "Yes": 2},
    "OnlineSecurity": {"No": 0, "No internet service": 1, "Yes": 2},
    "OnlineBackup": {"No": 0, "No internet service": 1, "Yes": 2},
    "DeviceProtection": {"No": 0, "No internet service": 1, "Yes": 2},
    "TechSupport": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingTV": {"No": 0, "No internet service": 1, "Yes": 2},
    "StreamingMovies": {"No": 0, "No internet service": 1, "Yes": 2},
}

one_hot_columns = ["Contract", "InternetService", "PaymentMethod"]

numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]


try:
    background_data = np.load("X_train_background.npy")
except FileNotFoundError:
    background_data = np.zeros((1, len(feature_names)))

explainer = shap.LinearExplainer(model, background_data)


def preprocess(customer_data):
    df = pd.DataFrame([customer_data])

    for col in label_encoded_columns:
        if col in df.columns:
            mapping = label_maps[col]
            df[col] = df[col].map(mapping)
            df[col] = df[col].fillna(0)  
            df[col] = df[col].astype(int)

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)


    df = pd.get_dummies(df, columns=one_hot_columns)

    df = df.reindex(columns=feature_names, fill_value=0)

    df[feature_names] = scaler.transform(df[feature_names])

    return df


@app.route("/predict", methods=["POST"])
def predict():
    customer_data = request.get_json()

    X = preprocess(customer_data)

    probabilities = model.predict_proba(X)
    churn_probability = probabilities[0][1]  # probability of "will churn"

    will_churn = churn_probability >= 0.5

    shap_values = explainer.shap_values(X)
    shap_values_for_this_customer = shap_values[0]  # first (only) row

    explanations = []
    for i in range(len(feature_names)):
        explanations.append({
            "feature": feature_names[i],
            "impact": float(shap_values_for_this_customer[i])
        })

    # Send the result back as JSON
    return jsonify({
        "churn_probability": round(float(churn_probability), 3),
        "will_churn": bool(will_churn),
        "reasons": explanations
    })

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000)