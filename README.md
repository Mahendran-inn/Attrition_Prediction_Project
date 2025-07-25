# 🧠 Employee Attrition Prediction

A machine learning-powered web app using **Logistic Regression** and **Streamlit** to predict whether an employee is likely to leave the organization (attrition). This app helps HR teams take proactive steps toward improving employee retention.

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

| Feature           | Description                          |
|------------------|--------------------------------------|
| Age              | Age of the employee                  |
| MonthlyIncome    | Employee salary                      |
| JobSatisfaction  | Job satisfaction rating (1 to 4)     |
| DistanceFromHome | Distance from home to office (km)    |
| OverTime         | Whether the employee works overtime  |
| BusinessTravel   | Frequency of business travel         |
| YearsAtCompany   | Total years spent in the company     |

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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('Employee-Attrition - Employee-Attrition.csv')

# Extract target variable
target = 'Attrition'
selected_features = [
    'Age', 'MonthlyIncome', 'Gender', 'MaritalStatus',
    'JobRole', 'YearsAtCompany', 'YearsInCurrentRole']

#spliting feature and label 
x = df[selected_features]
y = df['Attrition'] 

# Encode categorical columns
encoders = {}
for column in ['Gender', 'MaritalStatus', 'JobRole']:
    encoders[column] = LabelEncoder()
    x[column] = encoders[column].fit_transform(x[column])

# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train logistic regression model
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

# Save model, encoders, and feature order
with open('Attrition_prediction.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('Attrition_encoder.pkl', 'wb') as f:
    pickle.dump(encoders, f)

with open('feature_order.pkl', 'wb') as f:
    pickle.dump(feature_order, f)

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

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="🧠")

st.title("Employee Attrition Predictor")
st.markdown("Use this app to predict whether an employee is likely to leave the company.")

etc.......still makes prediction

```

---

## 🛠 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/employee-attrition-app.git
cd employee-attrition-app
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


