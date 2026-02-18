
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

# ---- Input fields ----
age = st.number_input("Age", min_value=18, max_value=100, value=30)

health_score = st.slider("Health Score", min_value=0, max_value=100, value=70)

previous_claims = st.number_input("Previous Claims", min_value=0, value=0)

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)

exercise_freq = st.selectbox(
    "Exercise Frequency",
    ["Rarely", "Monthly", "Weekly", "Daily"]
)

# ---- Encode Exercise Frequency (same as training) ----
exercise_map = {
    "Rarely": 1,
    "Monthly": 2,
    "Weekly": 3,
    "Daily": 4
}

exercise_encoded = exercise_map[exercise_freq]

#premium amount prediction 
if st.button("Predict Premium"):

    # IMPORTANT: column names must match training
    input_df = pd.DataFrame([{
        "Age": age,
        "Health Score": health_score,
        "Previous Claims": previous_claims,
        "Credit Score": credit_score,
        "Exercise Frequency Encoded": exercise_encoded
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Insurance Premium: ₹ {prediction:,.2f}")




