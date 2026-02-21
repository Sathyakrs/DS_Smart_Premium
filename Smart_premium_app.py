import os
import mlflow
import streamlit as st
import pandas as pd
import numpy as np


# STREAMLIT PAGE CONFIG

st.set_page_config(page_title="Smart Premium Predictor", page_icon="🛡️")


# SET DAGSHUB CREDENTIALS

os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["DAGSHUB_USER"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["DAGSHUB_TOKEN"]


# SET TRACKING URI

mlflow.set_tracking_uri("https://dagshub.com/zsathya103/Ds_Smart_premium.mlflow")


# LOAD PRODUCTION MODEL

@st.cache_resource
def load_model():
    return mlflow.pyfunc.load_model(
        "models:/SmartPremium_XGBoost/Production"
    )

model = load_model()

# STREAMLIT UI
st.title("🛡️ Smart Insurance Premium Prediction")
st.write("Enter customer details to estimate insurance premium.")

# INPUTS

age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=0.0, value=500000.0)
dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
health_score = st.slider("Health Score", 1, 100, 70)
previous_claims = st.number_input("Previous Claims", min_value=0, max_value=20, value=0)
vehicle_age = st.number_input("Vehicle Age", min_value=0, max_value=30, value=5)
credit_score = st.slider("Credit Score", 300, 900, 650)
insurance_duration = st.number_input("Insurance Duration (Years)", min_value=1, max_value=30, value=5)

gender = st.selectbox("Gender", ["Female", "Male"])
marital = st.selectbox("Marital Status", ["Single", "Married"])
policy = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
location = st.selectbox("Location", ["Rural", "Suburban", "Urban"])
property_type = st.selectbox("Property Type", ["Apartment", "Condo", "House"])

education_encoded = st.number_input("Education Level Encoded", min_value=0, max_value=5, value=1)
exercise_encoded = st.number_input("Exercise Frequency Encoded", min_value=0, max_value=5, value=2)
smoking_encoded = st.number_input("Smoking Status Encoded", min_value=0, max_value=1, value=0)

# PREDICTION BUTTON 

if st.button("Predict Premium"):

    input_data = {
        "Age": age,
        "Annual Income": income,
        "Number of Dependents": dependents,
        "Health Score": health_score,
        "Previous Claims": previous_claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit_score,
        "Insurance Duration": insurance_duration,
        "Education Level Encoded": education_encoded,
        "Exercise Frequency Encoded": exercise_encoded,
        "Smoking Status Encoded": smoking_encoded,
    }

    # One-hot encoding
    input_data["Gender_Male"] = 1 if gender == "Male" else 0
    input_data["Marital Status_Married"] = 1 if marital == "Married" else 0
    input_data["Marital Status_Single"] = 1 if marital == "Single" else 0
    input_data["Policy Type_Comprehensive"] = 1 if policy == "Comprehensive" else 0
    input_data["Policy Type_Premium"] = 1 if policy == "Premium" else 0
    input_data["Location_Suburban"] = 1 if location == "Suburban" else 0
    input_data["Location_Urban"] = 1 if location == "Urban" else 0
    input_data["Property Type_Condo"] = 1 if property_type == "Condo" else 0
    input_data["Property Type_House"] = 1 if property_type == "House" else 0

    expected_columns = [
        'Age', 'Annual Income', 'Number of Dependents',
        'Health Score', 'Previous Claims', 'Vehicle Age',
        'Credit Score', 'Insurance Duration',
        'Education Level Encoded', 'Exercise Frequency Encoded',
        'Smoking Status Encoded', 'Gender_Male',
        'Marital Status_Married', 'Marital Status_Single',
        'Policy Type_Comprehensive', 'Policy Type_Premium',
        'Location_Suburban', 'Location_Urban',
        'Property Type_Condo', 'Property Type_House'
    ]

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Premium: ₹ {prediction:,.2f}")
