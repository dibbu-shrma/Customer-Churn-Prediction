# 📊 ChurnSense – Customer Churn Prediction System

An end-to-end Machine Learning web application that predicts customer churn and explains each prediction using SHAP (SHapley Additive exPlanations).

Built using Flask, Scikit-learn, and a custom HTML/CSS/JavaScript frontend, ChurnSense demonstrates the complete ML workflow—from data preprocessing and model selection to deployment.

---

## 🚀 Live Demo

🔗 https://churnsense-778k.onrender.com

---

## ✨ Features

- Predicts whether a customer is likely to churn
- Displays churn probability
- Explains predictions using SHAP feature contributions
- Modern responsive dark-themed interface
- End-to-end ML deployment with Flask
- Trained model with preprocessing pipeline identical to training

---

## 📂 Dataset

**Telco Customer Churn Dataset**

- Source: Kaggle
- Records: 7,043 customers
- Features: 19 customer, service and billing attributes
- Task: Binary Classification (Churn / No Churn)

---

## 🧹 Data Preprocessing

The dataset was cleaned and transformed before training.

### Steps performed

- Converted `TotalCharges` from string to numeric
- Handled missing values
- Encoded binary categorical variables
- Applied One-Hot Encoding to multi-category features
- Standardized numerical features using `StandardScaler`

The exact preprocessing pipeline is reproduced during inference to ensure consistent predictions.

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

Hyperparameter tuning was performed using **GridSearchCV**.

### 🏆 Final Model

**Logistic Regression**

Best Parameters:

```python
C = 10
penalty = "l1"
solver = "liblinear"
max_iter = 1000
```

The trained model, scaler and feature metadata are stored as:

- model.pkl
- scaler.pkl
- feature_names.json

---

## 🔍 Explainable AI

Instead of only predicting churn, ChurnSense explains **why** a prediction was made.

Using **SHAP LinearExplainer**, the application highlights the features that increase or decrease churn probability for every prediction.

---

## 🛠 Tech Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP

### Backend

- Flask

### Frontend

- HTML
- CSS
- JavaScript

### Deployment

- Render

### Development Tools

- Jupyter Notebook
- Git
- GitHub
- VS Code

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── app.py
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── feature_names.json
├── templates/
├── static/
├── notebook/
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
```

Move into the project folder

```bash
cd customer-churn-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000
```

---

## 📸 Application Preview

### Dashboard

<img width="1620" height="950" alt="Screenshot 2026-07-05 182528" src="https://github.com/user-attachments/assets/722e472c-20b3-4df9-8476-61d561f10095" />

### Risk Prediction

<img width="1900" height="962" alt="Screenshot 2026-07-05 182617" src="https://github.com/user-attachments/assets/19170a2c-32f7-49a7-9e4a-f0f352e72772" />

### SHAP Explanation

<img width="1493" height="972" alt="Screenshot 2026-07-05 182704" src="https://github.com/user-attachments/assets/36f7d34e-674d-44d1-9934-c943282cac7a" />

### Model Comparison

<img width="1920" height="957" alt="new" src="https://github.com/user-attachments/assets/b02c4839-a851-43d4-87f8-8242fe659141" />

---

## 📚 What I Learned

This project helped me gain hands-on experience with:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Model comparison and evaluation
- Hyperparameter tuning using GridSearchCV
- Building REST APIs with Flask
- Explainable AI using SHAP
- Integrating Machine Learning with a custom frontend
- Deploying ML applications using Render
- Version control with Git and GitHub

---

## ⚠️ Disclaimer

This application was developed for educational and portfolio purposes.

Predictions are based on historical data and should not be used as the sole basis for real-world business decisions.

---

## 👩‍💻 Author

**Divya Sharma**

If you found this project interesting, feel free to connect with me on LinkedIn or explore the repository.
