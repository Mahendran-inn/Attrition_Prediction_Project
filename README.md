# 🧠 Employee Attrition Prediction

A machine learning-powered web using **Logistic Regression** and **Streamlit** to predict whether an employee is likely to leave the organization (attrition). This helps HR teams take proactive steps toward improving employee retention.

---

## 📌 Table of Contents

- [🔍 Problem Statement](#-problem-statement)
- [🗃️ Dataset Overview](#-dataset-overview)
- [🧠 Machine Learning Model](#-machine-learning-model)
- [🧪 Model Training Script](#-model-training-script)
- [🌐 Streamlit Web App Script](#-streamlit-web-app-script)
- [🛠 Setup Instructions](#-setup-instructions)
- [📁 Project Structure](#-project-structure)
- [🚀 How to Run the App](#-how-to-run-the-app)
- [🎯 Future Enhancements](#-future-enhancements)


---

## 🔍 Problem Statement

Employee attrition can have a major impact on an organization's performance and costs. The objective of this project is to build a model that can predict whether an employee is likely to leave the organization, based on historical data.

---

## 🗃️ Dataset Overview

- Dataset: `Employee-Attrition - Employee-Attrition.csv`
- Target Variable: `Attrition` (Yes = employee will leave, No = employee will stay)

**Example Features:**

| Feature            | Description                          |
|--------------------|--------------------------------------|
| Age                | Age of the employee                  |
| MonthlyIncome      | Employee salary                      |
| Gender             | Gender selection (male or female)    |
| MaritalStatus      | MaritalStatus (single or married)    |
| JobRole            | Employee's job role                  |
| YearsInCurrentRole | Total years spent in the currentrole |
| YearsAtCompany     | Total years spent in the company     |

---

## 🧠 Machine Learning Model

- Algorithm: **Logistic Regression**
- Data Preprocessing:
  - Label Encoding for categorical features
  - Save encoder dictionary and feature order
- Model Evaluation:
  - Trained using `train_test_split` (80/20)
  - Saved using `pickle`

---

## 🧪 Model Training Script

```python

import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
# Features & target
attrition_features = [
    'Age', 'MonthlyIncome', 'Gender', 'MaritalStatus',
    'JobRole', 'YearsAtCompany', 'YearsInCurrentRole'
]
attrition_target = 'Attrition'

# Drop NA
attrition_df = data[attrition_features + [attrition_target]].dropna()

# Encode categorical
attrition_encoders = {}
for col in ['Gender', 'MaritalStatus', 'JobRole']:
    encoder = LabelEncoder()
    attrition_df[col] = encoder.fit_transform(attrition_df[col])
    attrition_encoders[col] = encoder

# Split features & labels
X_attrition = attrition_df[attrition_features]
y_attrition = attrition_df[attrition_target]

from sklearn.model_selection import train_test_split
# Train-test split
X_attr_train, X_attr_test, y_attr_train, y_attr_test = train_test_split(
    X_attrition, y_attrition, test_size=0.2, random_state=1
)


from sklearn.linear_model import LogisticRegression
# Model training
attr_model = LogisticRegression(max_iter=1000)
attr_model.fit(X_attr_train, y_attr_train)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Prediction
y_attr_pred = attr_model.predict(X_attr_test)

# Error Metrics
print("\nAttrition Prediction Metrics:")
print("Accuracy:", accuracy_score(y_attr_test, y_attr_pred))
print("Classification Report:\n", classification_report(y_attr_test, y_attr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_attr_test, y_attr_pred))

# Save model & encoders
with open("Attrition_model.pkl", "wb") as f:
    pickle.dump(attr_model, f)

with open("Attrition_encoder.pkl", "wb") as f:
    pickle.dump(attrition_encoders, f)

print("✅ Model training complete and files saved!")
````

---

## 🌐 Streamlit Web App Script

```python

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load saved model and encoders
Attrition_model = pickle.load(open('model.pkl', 'rb'))
Attrition_encoder = pickle.load(open('encoders.pkl', 'rb'))
feature_order = pickle.load(open('feature_order.pkl', 'rb'))

st.set_page_config(page_title="Employee Attrition & Job Satisfaction")
st.title("Employee Attrition & Job Satisfaction Predictor")
st.markdown("Use this app to predict **Attrition likelihood** and **Job Satisfaction level** of employees.")

etc.......still makes prediction

```

---

## 🛠 Setup Instructions

### 1. New Repository

```bash
[git clone https://github.com/yourusername/employee-attrition-app.git
cd employee-attrition-app](http://github.com/Mahendran-inn/Attrition_Prediction_Project/tree/main)
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

`requirements.txt` content:

```
pandas
numpy
scikit-learn
streamlit
```

### 3. Train the Model

```bash
python model_train.py
```

### 4. Run the App

```bash
streamlit run app.py
```

Then open your browser and go to: [http://localhost:8501](http://localhost:8501)

---

## 📁 Project Structure

```
employee-attrition/
│
├── Employee-Attrition.csv        # Dataset
├── model_train.py                # Script to train the model
├── app.py                        # Streamlit application
├── model.pkl                     # Trained model
├── encoders.pkl                  # Label encoders
├── feature_order.pkl             # Column order for prediction
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## 🚀 How to Run the App

```bash
streamlit run app.py
```

---

## 🎯 Future Enhancements

* Add more models (Random Forest, XGBoost)
* Add model evaluation metrics
* Explain predictions using SHAP
* Enable batch upload (CSV file prediction)
* Deploy to Streamlit Cloud or HuggingFace Spaces

---


