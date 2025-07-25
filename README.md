# 🧠 Employee Attrition Prediction

A machine learning-powered web application built with **Streamlit** to predict whether an employee is likely to leave the company. This solution empowers HR departments to proactively understand attrition risks and act accordingly.

---

## 📌 Table of Contents

- [🔍 Problem Statement](#-problem-statement)
- [🗃️ Dataset Description](#-dataset-description)
- [🧪 Project Workflow](#-project-workflow)
- [🔧 Installation & Setup](#-installation--setup)
- [🚀 How to Run](#-how-to-run)
- [📊 Model Building](#-model-building)
- [🌐 Streamlit Web App](#-streamlit-web-app)
- [📁 Project Structure](#-project-structure)

---

## 🔍 Problem Statement

High employee turnover can be expensive and disruptive. This project aims to:

- Predict employee attrition using historical data.
- Provide HR teams with actionable insights.
- Offer a user-friendly interface for real-time predictions.

---

## 🗃️ Dataset Description

Dataset: `Employee-Attrition - Employee-Attrition.csv`

Key features include:

| Column            | Description                                |
|------------------|--------------------------------------------|
| Age              | Employee age                                |
| Department       | HR / Sales / R&D                            |
| DistanceFromHome | Distance in kilometers                      |
| MonthlyIncome    | Employee's salary                           |
| OverTime         | Whether employee works overtime             |
| JobSatisfaction  | Satisfaction level (1 to 4)                 |
| YearsAtCompany   | Tenure with the company                     |
| Attrition        | **Target**: Yes (left) / No (still working) |

Categorical variables are encoded using `LabelEncoder`. Feature order is preserved using `feature_order.pkl`.

---

## 🧪 Project Workflow

```text
📦 Load Dataset
🧹 Clean and Encode Data
🧠 Train Logistic Regression Model
📦 Save Model with Pickle
💻 Build Web App with Streamlit
🚀 Launch Locally or Deploy

---

## 🔧 Installation & Setup

1.Clone the repository

git clone https://github.com/yourusername/employee-attrition-app.git
cd employee-attrition-app

2.Install dependencies
pip install -r requirements.txt


3.🚀 How to Run the App
streamlit run app.py

Then open your browser to: http://localhost:8501

---

##📊 Model Building
✅ Data Preprocessing
Null/missing value check

Label encoding of categorical columns

Standard feature scaling (optional)

✅ Model Training
Using Logistic Regression from scikit-learn:

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

Saved with:

import pickle
with open('Attrition_prediction.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

Also saved:

encoders.pkl: Label encoders dictionary

feature_order.pkl: Ensures correct input ordering in the app


##🌐 Streamlit Web App Features

Sidebar input widgets for employee features

Real-time prediction: "Will Leave" / "Will Stay"

Simple, clean interface

Input validation and feedback

---

##📁 Project Structure

employee-attrition-app/
│
├── app.py                  # Streamlit web app
├── Employee-Attrition.csv  # Dataset
├── model.pkl               # Trained Logistic Regression model
├── encoders.pkl            # Label encoders
├── feature_order.pkl       # List of model input feature order
├── requirements.txt        # Python dependencies
└── README.md               # You're reading it!

---

##🥇 Credits
Built by: [Mahendran]

Tools Used: Python, Scikit-learn, Streamlit
---



