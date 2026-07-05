import pickle
import json
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# -----------------------------------------------------------------
# STEP 1: Load everything we saved from training, once, at startup
# -----------------------------------------------------------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

# -----------------------------------------------------------------
# STEP 2: Write down exactly how each column was encoded in training
# -----------------------------------------------------------------

# These columns were label-encoded (turned into numbers like 0, 1, 2)
label_encoded_columns = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "PaperlessBilling"
]

# The actual number each text value maps to (alphabetical order,
# same as what LabelEncoder does automatically)
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

# These columns were one-hot encoded with pd.get_dummies
one_hot_columns = ["Contract", "InternetService", "PaymentMethod"]

# These columns were scaled with StandardScaler
numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

# -----------------------------------------------------------------
# STEP 3: Set up SHAP so we can explain each prediction
# -----------------------------------------------------------------

# SHAP needs a "background" to compare each prediction against.
# If you have a saved sample of real training data, we use that.
# Otherwise we just use a row of zeros as a simple placeholder.
try:
    background_data = np.load("X_train_background.npy")
except FileNotFoundError:
    background_data = np.zeros((1, len(feature_names)))

explainer = shap.LinearExplainer(model, background_data)


# -----------------------------------------------------------------
# STEP 4: The preprocessing function
# This just repeats, step by step, what you did in your notebook.
# -----------------------------------------------------------------

def preprocess(customer_data):
    # Turn the single customer dict into a one-row DataFrame
    df = pd.DataFrame([customer_data])

    # Convert label-encoded columns (like "Yes"/"No") into numbers
    for col in label_encoded_columns:
        if col in df.columns:
            mapping = label_maps[col]
            df[col] = df[col].map(mapping)
            df[col] = df[col].fillna(0)  # just in case of an unexpected value
            df[col] = df[col].astype(int)

    # SeniorCitizen is already sent as "0"/"1" string from the form — cast to int
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)


    # One-hot encode the multi-category columns
    df = pd.get_dummies(df, columns=one_hot_columns)

    # Make sure the columns match training exactly (same order,
    # same set of columns). Anything missing gets filled with 0.
    df = df.reindex(columns=feature_names, fill_value=0)

    # Scale using the SAME scaler from training
    # (scaler.pkl was fit on the full 26-feature dataframe, not just the numeric columns)
    df[feature_names] = scaler.transform(df[feature_names])

    return df


# -----------------------------------------------------------------
# STEP 5: The actual /predict route
# -----------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    # Get the JSON the frontend sent us
    customer_data = request.get_json()

    # Preprocess it exactly like training data
    X = preprocess(customer_data)

    # Ask the model for a churn probability
    probabilities = model.predict_proba(X)
    churn_probability = probabilities[0][1]  # probability of "will churn"

    # Turn that into a simple yes/no prediction
    will_churn = churn_probability >= 0.5

    # Get SHAP values to explain WHY the model predicted this
    shap_values = explainer.shap_values(X)
    shap_values_for_this_customer = shap_values[0]  # first (only) row

    # Pair each feature name with its SHAP value
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