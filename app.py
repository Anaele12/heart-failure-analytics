import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set up page configurations
st.set_page_config(
    page_title="Heart Failure Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Heart Failure Predictive Analytics Dashboard")
st.write("Examine patient clinical records and evaluate cardiovascular risk using machine learning.")

# --- STEP 1: SAFE MODEL & SCALER LOADING ---
try:
    loaded_scaler = joblib.load('scaler.pkl')
    loaded_model = joblib.load('heart_model.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}. Please ensure 'scaler.pkl' and 'heart_model.pkl' are in your root repository folder.")

# --- STEP 2: SIDEBAR USER INPUTS ---
st.sidebar.header("Patient Clinical Metrics")

age_input = st.sidebar.slider("Age", min_value=1, max_value=120, value=50)
sex_input = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", options=[0, 1])
resting_bp_input = st.sidebar.slider("Resting Blood Pressure (mm Hg)", min_value=50, max_value=220, value=120)
cholesterol_input = st.sidebar.slider("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
fasting_bs_input = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)", options=[0, 1])
max_hr_input = st.sidebar.slider("Maximum Heart Rate Achieved (bpm)", min_value=50, max_value=250, value=150)
exercise_angina_input = st.sidebar.selectbox("Exercise-Induced Angina (0 = No, 1 = Yes)", options=[0, 1])
oldpeak_input = st.sidebar.slider("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
st_slope_input = st.sidebar.selectbox("ST Slope Type (1 = Upsloping, 2 = Flat, 3 = Downsloping)", options=[1, 2, 3])

chest_pain = st.sidebar.selectbox("Chest Pain Type", options=["ASY", "ATA", "NAP", "TA"])
resting_ecg = st.sidebar.selectbox("Resting ECG Results", options=["Normal", "LVH", "ST"])

# --- STEP 3: FEATURE ALIGNMENT AND PROCESSING ---
# We manually structure the dictionary to match the exact uppercase/lowercase layouts 
# seen during your model's training phase.
input_data = {
    'Age': age_input,
    'Sex': sex_input,
    'RestingBP': resting_bp_input,
    'Cholesterol': cholesterol_input,
    'FastingBS': fasting_bs_input,
    'MaxHR': max_hr_input,
    'ExerciseAngina': exercise_angina_input,
    'Oldpeak': oldpeak_input,
    'ST_Slope': st_slope_input,
    
    # Matching exact fit-time cases ('_Ata', '_Nap', '_Ta')
    'ChestPainType_Ata': 1 if chest_pain == 'ATA' else 0,
    'ChestPainType_Nap': 1 if chest_pain == 'NAP' else 0,
    'ChestPainType_Ta': 1 if chest_pain == 'TA' else 0,
    
    # Matching exact fit-time cases ('_LVH', '_Normal', '_ST')
    'RestingECG_LVH': 1 if resting_ecg == 'LVH' else 0,
    'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
    'RestingECG_ST': 1 if resting_ecg == 'ST' else 0
}

# Convert dictionary to DataFrame
input_df = pd.DataFrame([input_data])

# Enforce the explicit column order required by scikit-learn
feature_order = [
    'Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 
    'ExerciseAngina', 'Oldpeak', 'ST_Slope', 
    'ChestPainType_Ata', 'ChestPainType_Nap', 'ChestPainType_Ta', 
    'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST'
]
input_df = input_df[feature_order]

# --- STEP 4: PREDICTION PIPELINE ---
st.write("### Patient Metrics Summary")
st.dataframe(input_df)

if st.button("Run Risk Assessment"):
    try:
        # Scale the features using the loaded scaler
        scaled_input = loaded_scaler.transform(input_df)
        
        # Pass the processed metrics to your classification model
        prediction = loaded_model.predict(scaled_input)
        prediction_proba = loaded_model.predict_proba(scaled_input)[0][1]
        
        st.write("---")
        st.write("### Assessment Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 1:
                st.error("⚠️ **High Risk Detected**")
                st.write("The model classifies this patient as possessing clinical markers highly correlated with heart failure risk.")
            else:
                st.success("✅ **Low Risk Detected**")
                st.write("The model classifies this patient as possessing clinical markers indicating lower cardiovascular vulnerability.")
                
        with col2:
            st.metric(label="Calculated Probability of Heart Failure", value=f"{prediction_proba * 100:.1f}%")
            st.progress(float(prediction_proba))
            
    except NameError:
        st.error("Model file loading error occurred. Could not run calculations.")
    except Exception as prediction_error:
        st.error(f"An unexpected error occurred during valuation: {prediction_error}")
