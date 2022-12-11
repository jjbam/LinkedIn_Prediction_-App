
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#title
st.set_page_config(page_title="LinkedIn User Prediction App")
st.image("LinkedIn_Logo.png")
st.markdown("Welcome to the LinkedIn User Analytics App. The goal of this app is to predict whether an individual is a LinkedIn User, while providing a percent prediction based on our Machine Learning Model.")

#take input for income
income = st.selectbox("What Is Your Income Range?",
            options = ["Less than $10,000", "10 to under $20,000", 
            "20 to under $30,000", "30 to under $40,000", "40 to under $50,000",
            "50 to under $75,000", "75 to under $100,000", "100 to under $150,000",
            "150,000 or more"])
#translate input for income
if income == "Less than $10,000":
    income = 1
elif income == "10 to under $20,000":
    income = 2 
elif income == "20 to under $30,000":
    income = 3
elif income == "30 to under $40,000":
    income = 4
elif income == "40 to under $50,000":
    income = 5
elif income == "50 to under $75,000":
    income = 6
elif income == "75 to under $100,000":
    income = 7
elif income == "100 to under $150,000":
    income = 8
else:
    income = 9

#take input for education
educ2 = st.selectbox("What Is Your Education Level?", 
            options = ["Less than high school (Grades 1-8 or no formal schooling)",
            "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)", 
            "High school graduate (Grade 12 with diploma or GED certificate)",
            "Some college, no degree (includes some community college)", 
            "Two-year associate degree from a college or university",
            "Four-year college or university degree/Bachelorâs degree (e.g., BS, BA, AB)",
            "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
            "Postgraduate or professional degree, including masterâs, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"])
#translate input for education
if educ2 == "Less than high school (Grades 1-8 or no formal schooling)":
    educ2 = 1
elif educ2 == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
    educ2 = 2 
elif educ2 == "High school graduate (Grade 12 with diploma or GED certificate)":
    educ2 = 3
elif educ2 == "Some college, no degree (includes some community college)":
    educ2 = 4
elif educ2 == "Two-year associate degree from a college or university":
    educ2 = 5
elif educ2 == "Four-year college or university degree/Bachelorâs degree (e.g., BS, BA, AB)":
    educ2 = 6
elif educ2 == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    educ2 = 7
else:
    educ2 = 8

#take input for parent
par = st.selectbox("Are You a Parent?",
options = ["Yes", "No"])
#Translate input for parent
if par == "Yes":
    par = 1
else:
    par = 2

#Take input for Marital
marital  = st.selectbox("Select Your Marital Status.",
            options = ["Married", "Living with a partner", "Divorced",
           "Separated", "Widowed", "Never been married"])
#Translate input for Marital
if marital == "Married":
    marital = 1
elif marital == "Living with a partner":
    marital = 2 
elif marital == "Divorced":
    marital = 3
elif marital == "Separated":
    marital = 4
elif marital == "Widowed":
    marital = 5
else:
    marital = 6

#Take input for gender
gender = st.selectbox("Select Your Gender.",
            options = ["Male", "Female", "Other"])
#Translate input for gender
if gender == "Male":
    gender = 1
elif gender == "Female":
    gender = 2
else:
    gender = 3

#Take input for Age
value = int
type(value)
age = st.slider("Enter Your Age.", min_value=16, max_value=97)
#Translate input for age
age = age
#Textblob to analyze input

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

ss = pd.DataFrame({
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] > 2, np.nan, 
                     np.where(s["par"] == 1, 1, 0)),
    "married":np.where(s["marital"] > 6, np.nan, 
                     np.where(s["marital"] == 1, 1, 0)),
    "female":np.where(s["gender"] > 3, np.nan, 
                     np.where(s["gender"] == 2, 1, 0)),
    "age":np.where(s["age"] > 97, np.nan, s["age"]),
    "sm_li":clean_sm(s["web1h"])})
ss = ss.dropna()

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y, #same number of taeget in training and test
                                                    test_size = .2, #set aside 20% of data for testing
                                                    random_state=850) #set 

lr = LogisticRegression()
#initialize algorithm
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

userinput = [income, educ2, par, marital, gender, age]
predicted_class = lr.predict([userinput])
probs = lr.predict_proba([userinput])

if predicted_class == 0:
    st.write("Classification: Not a LinkedIn User")
else:
    st.write("LinkedIn User") # 0=not LinkedIn user, 1=LinkedIn user
#if(f"Predicted classification: {predicted_class[0]}")
st.write(f"Probability that this person is a LinkedIn user: {probs[0][1]}")

