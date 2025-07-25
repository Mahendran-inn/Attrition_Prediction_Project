import pandas as pd

data=pd.read_csv('Employee-Attrition - Employee-Attrition.csv')
df=data.copy()

#df.isnull().mean()*100
#df.describe()
#df.describe().columns


#To find the outliers
def outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

numerical_column=['Age', 'DailyRate', 'DistanceFromHome', 'Education',
       'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
       'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
       'StockOptionLevel', 'TotalWorkingYears','WorkLifeBalance', 
       'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']

for col in numerical_column:
    outliers = outliers_iqr(df, col)
    print(f"Outliers in {col}: {len(outliers)}")
    #sns.boxplot(x=df[col])
    #plt.title(f'Boxplot for {col}')
    #plt.show()
    
#Scale selected columns (RobustScaler)
from sklearn.preprocessing import RobustScaler
columns_to_treat = [
    'MonthlyIncome','NumCompaniesWorked','PerformanceRating', 'StockOptionLevel',
    'TotalWorkingYears','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion', 
    'YearsWithCurrManager'
    ]

scaler = RobustScaler()
df[columns_to_treat] = scaler.fit_transform(df[columns_to_treat])


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#EDA:

# Monthly Income vs Attrition
sns.scatterplot(x='Attrition', y='MonthlyIncome', data=df)

# Monthly Income vs Attrition
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)


# Distance from Home by Attrition
sns.boxplot(x='Attrition', y='DistanceFromHome', data=df)

# Years at Company by Department
sns.boxplot(x='Department', y='YearsAtCompany', data=df)

# Age by OverTime
sns.boxplot(x='OverTime', y='Age', data=df)

#Target Variable Distribution
sns.countplot(x='Attrition', data=df)
plt.title('Attrition Count (0 = Stay, 1 = Leave)')
plt.show()

#Univariate Analysis
## Age
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# JobSatisfaction
sns.countplot(x='JobSatisfaction', data=df)
plt.title('Job Satisfaction Distribution')
plt.show()

## Gender
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.show()

# Bivariate Analysis
## Attrition by Gender
sns.countplot(x='Gender', hue='Attrition', data=df)
plt.title('Attrition by Gender')
plt.show()


## Attrition by Department
sns.countplot(x='Department', hue='Attrition', data=df)
plt.title('Attrition by Department')
plt.show()


## MonthlyIncome vs Attrition
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title('Monthly Income by Attrition')
plt.show()

selected_features = [
    'Age', 'MonthlyIncome', 'Gender', 'MaritalStatus',
    'JobRole', 'YearsAtCompany', 'YearsInCurrentRole']

#spliting feature and label 
x = df[selected_features]
y = df['Attrition'] 

#Encoding the categorical columns
from sklearn.preprocessing import LabelEncoder

encoders={}

for column in ['Gender', 'MaritalStatus', 'JobRole']:
    encoders[column] = LabelEncoder()
    x[column] = encoders[column].fit_transform(x[column])
    
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

#split for trainig the dataset
from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression().fit(x_train,y_train)
print(lr_model)


#Error metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = lr_model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


#optional model
#from sklearn.ensemble import RandomForestClassifier
#rf= RandomForestClassifier(n_estimators=100, random_state=1)

#rf.fit(x_train, y_train)
#y_pred = rf.predict(x_test)


#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("\nClassification Report:\n", classification_report(y_test, y_pred))
#print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))




import pickle
# To Save the model and Encoder(saved in dictionary)
with open('Attrition_prediction.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('Attrition_encoder.pkl', 'wb') as f:
    pickle.dump(encoders, f)


# To Load the model and Encoder(saved in dictionary)
with open('Attrition_prediction.pkl', 'rb') as f:
    Attrition_model = pickle.load(f)

with open('Attrition_encoder.pkl', 'rb') as f:
    Attrition_encoder = pickle.load(f)

#To save & load the selected_feature list
pickle.dump(selected_features, open("feature_order.pkl", "wb"))



import streamlit as st
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="🧠")
st.title(" Employee Attrition Predictor")
st.markdown("Use this app to predict whether an employee is likely to leave the company.")


with st.form("input_form"):
    st.subheader("Enter Input Data")

    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    jobrole = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ])
    income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    years_in_role = st.slider("Years in Current Role", 0, 20, 2)

    submit = st.form_submit_button("Predict")

if submit:
    try:
        # Encoding categorical values
        gender_encoded = Attrition_encoder['Gender'].transform([gender])[0]
        jobrole_encoded = Attrition_encoder['JobRole'].transform([jobrole])[0]
        marital_encoded = Attrition_encoder['MaritalStatus'].transform([marital_status])[0]

        # Prepare input as dictionary
        input_dict = {
            'Age': age,
            'Gender': gender_encoded,
            'JobRole': jobrole_encoded,
            'MonthlyIncome': income,
            'MaritalStatus': marital_encoded,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_role
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])[selected_features]

        # Predict
        prediction = Attrition_model.predict(input_df)[0]

        # Output result
        if prediction == 1:
            st.success(" Employee is likely to leave")
        else:
            st.success(" Employee is likely to stay")

    except Exception as e:
        st.error(f" Error during prediction: {e}")

