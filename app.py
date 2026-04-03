import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_excel('gender.xlsx')

# -------------------------------
# CLEAN DATA
# -------------------------------
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

df['occupation'] = df['occupation'].replace('?', df['occupation'].mode()[0])
df['workclass'] = df['workclass'].replace('?', df['workclass'].mode()[0])

# -------------------------------
# SELECT COLUMNS
# -------------------------------
df = df[['age','sex','education','occupation','workclass','hours.per.week','income']]

# -------------------------------
# ENCODING
# -------------------------------
le_sex = LabelEncoder()
le_edu = LabelEncoder()
le_occ = LabelEncoder()
le_work = LabelEncoder()
le_income = LabelEncoder()

df['sex'] = le_sex.fit_transform(df['sex'])
df['education'] = le_edu.fit_transform(df['education'])
df['occupation'] = le_occ.fit_transform(df['occupation'])
df['workclass'] = le_work.fit_transform(df['workclass'])
df['income'] = le_income.fit_transform(df['income'])

# -------------------------------
# SPLIT DATA
# -------------------------------
X = df[['age','sex','education','occupation','workclass','hours.per.week']]
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("Gender Inequality Prediction App")

st.write("Predict income based on gender, education and job details")

# INPUTS
age = st.slider("Age", 18, 90)

sex = st.selectbox("Gender", le_sex.classes_)
education = st.selectbox("Education", le_edu.classes_)
occupation = st.selectbox("Occupation", le_occ.classes_)
workclass = st.selectbox("Workclass", le_work.classes_)

hours = st.slider("Hours per week", 1, 100)

# ENCODE INPUT
sex_val = le_sex.transform([sex])[0]
edu_val = le_edu.transform([education])[0]
occ_val = le_occ.transform([occupation])[0]
work_val = le_work.transform([workclass])[0]

# PREDICT
if st.button("Predict"):
    input_data = np.array([[age, sex_val, edu_val, occ_val, work_val, hours]])
    prediction = model.predict(input_data)

    result = le_income.inverse_transform(prediction)[0]

    st.success(f"Predicted Income: {result}")