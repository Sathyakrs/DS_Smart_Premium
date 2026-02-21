
import os
import dagshub
import mlflow.pyfunc
import streamlit as st
import pandas as pd

#streamlit cloud uses sceret tokens
# Set credentials from Streamlit secrets
os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["DAGSHUB_USER"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["DAGSHUB_TOKEN"]

# Set DAGsHub tracking URI
mlflow.set_tracking_uri("https://dagshub.com/zsathya103/Ds_Smart_premium.mlflow")

#load production model
model = mlflow.pyfunc.load_model("models:/SmartPremium_XGBoost/Production")


#streamlit UI

st.set_page_config(page_title="Smart Premium Predictor", page_icon="🛡️")

st.title("🛡️ Smart Insurance Premium Prediction")
st.write("Enter customer details to estimate insurance premium.")

# User Inputs

age = st.number_input("Age")
income = st.number_input("Annual Income")
health_score = st.number_input("Health Score")
previous_claims = st.number_input("Previous Claims")
vehicle_age = st.number_input("Vehicle Age")
credit_score = st.number_input("Credit Score")
insurance_duration = st.number_input("Insurance Duration")
dependents = st.number_input("Number of Dependents")

education_encoded = st.number_input("Education Level Encoded")
exercise_encoded = st.number_input("Exercise Frequency Encoded")
smoking_encoded = st.number_input("Smoking Status Encoded")

gender = st.selectbox("Gender", ["Male", "Female"])
marital = st.selectbox("Marital Status", ["Single", "Married"])
policy = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
location = st.selectbox("Location", ["Rural", "Suburban", "Urban"])
property_type = st.selectbox("Property Type", ["Apartment", "Condo", "House"])

if st.button("Predict Premium"):

    input_df = pd.DataFrame([{
        "Age": age,
        "Annual Income": income,
        "Health Score": health_score,
        "Previous Claims": previous_claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit_score,
        "Insurance Duration": insurance_duration,
        "Number of Dependents": dependents,
        "Education Level Encoded": education_encoded,
        "Exercise Frequency Encoded": exercise_encoded,
        "Smoking Status Encoded": smoking_encoded,
        "Gender": gender,
        "Marital Status": marital,
        "Policy Type": policy,
        "Location": location,
        "Property Type": property_type
    }])

    prediction_log = model.predict(input_df)[0]
    prediction = np.expm1(prediction_log)

    st.success(f"Predicted Premium: ₹ {prediction:,.2f}")

