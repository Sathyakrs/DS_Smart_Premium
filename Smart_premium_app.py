
import os
import dagshub
import mlflow.pyfunc
import streamlit as st
import pandas as pd
import numpy as np

#streamlit cloud uses secret tokens
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

if st.button("Predict Premium"):

    # Base numeric features
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

    # Gender (Female is base)
    input_data["Gender_Male"] = 1 if gender == "Male" else 0

    # Marital Status (base = first category dropped during training)
    input_data["Marital Status_Married"] = 1 if marital == "Married" else 0
    input_data["Marital Status_Single"] = 1 if marital == "Single" else 0

    # Policy Type
    input_data["Policy Type_Comprehensive"] = 1 if policy == "Comprehensive" else 0
    input_data["Policy Type_Premium"] = 1 if policy == "Premium" else 0

    # Location
    input_data["Location_Suburban"] = 1 if location == "Suburban" else 0
    input_data["Location_Urban"] = 1 if location == "Urban" else 0

    # Property Type
    input_data["Property Type_Condo"] = 1 if property_type == "Condo" else 0
    input_data["Property Type_House"] = 1 if property_type == "House" else 0

    # Create dataframe
    input_df = pd.DataFrame([input_data])

    # IMPORTANT: reorder columns EXACTLY like training
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

    input_df = input_df[expected_columns]

    prediction_log = model.predict(input_df)[0]
    prediction = np.expm1(prediction_log)

    st.success(f"Predicted Premium: ₹ {prediction:,.2f}")