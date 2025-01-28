import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title of the app
st.title("Stroke Prediction App")

# Sidebar for user input
st.sidebar.header("User Input Features")

# Function to get user input
def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    age = st.sidebar.slider("Age", 0, 100, 30)
    
    # Change hypertension and heart disease to 'Yes'/'No'
    hypertension = st.sidebar.selectbox("Hypertension", ("No", "Yes"))
    heart_disease = st.sidebar.selectbox("Heart Disease", ("No", "Yes"))
    
    ever_married = st.sidebar.selectbox("Ever Married", ("Yes", "No"))
    work_type = st.sidebar.selectbox("Work Type", 
                                     ("Private", "Self-employed", "Govt_job", "Children", "Never_worked"))
    residence_type = st.sidebar.selectbox("Residence Type", ("Urban", "Rural"))
    avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
    smoking_status = st.sidebar.selectbox("Smoking Status", 
                                          ("formerly smoked", "never smoked", "smokes"))
    
    # Map categorical inputs to numerical values as per model preprocessing
    gender = 1 if gender == "Male" else 0
    ever_married = 1 if ever_married == "Yes" else 0
    work_type_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "Children": 3, "Never_worked": 4}
    residence_type = 1 if residence_type == "Urban" else 0
    smoking_status_map = {"formerly smoked": 0, "never smoked": 1, "smokes": 2}
    
    # Map hypertension and heart disease
    hypertension = 1 if hypertension == "Yes" else 0
    heart_disease = 1 if heart_disease == "Yes" else 0
    
    work_type = work_type_map[work_type]
    smoking_status = smoking_status_map[smoking_status]

    # Combine inputs into a single array
    data = np.array([gender, age, hypertension, heart_disease, ever_married,
                     work_type, residence_type, avg_glucose_level, bmi, smoking_status]).reshape(1, -1)
    
    return data

# Get user input
input_data = user_input_features()

# Predict stroke
if st.button("Predict"):
    prediction_prob = model.predict_proba(input_data)[0][1]  # Probability of stroke

    # Determine the risk level based on custom thresholds
    if prediction_prob < 0.15:
        risk_level = "✅ Low risk of stroke."
    elif 0.15 <= prediction_prob < 0.45:
        risk_level = "⚠️ Medium risk of stroke."
    else:
        risk_level = "⚠️ High risk of stroke!"

    # Display results
    st.write(f"### Prediction: {risk_level}")
    st.write(f"Probability of stroke: **{prediction_prob:.2f}**")
