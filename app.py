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
sex_input = st.sidebar.selectbox("Sex", options=["M", "F"])
resting_bp_input = st.sidebar.slider("Resting Blood Pressure (mm Hg)", min_value=50, max_value=220, value=120)
cholesterol_input = st.sidebar.slider("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
fasting_bs_input = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)", options=[0, 1])
max_hr_input = st.sidebar.slider("Maximum Heart Rate Achieved (bpm)", min_value=50, max_value=250, value=150)
exercise_angina_input = st.sidebar.selectbox("Exercise-Induced Angina (0 = No, 1 = Yes)", options=[0, 1])
oldpeak_input = st.sidebar.slider("Oldpeak (ST depression induced by exercise)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
st_slope_input = st.sidebar.selectbox("ST Slope Type", options=["Up", "Flat", "Down"])

chest_pain = st.sidebar.selectbox("Chest Pain Type", options=["ASY", "ATA", "NAP", "TA"])
resting_ecg = st.sidebar.selectbox("Resting ECG Results", options=["Normal", "LVH", "ST"])

# --- STEP 3: FEATURE ENGINEERING & ONE-HOT CODES ---

# 1. Compute your data science engineered features
cardiac_workload = resting_bp_input * max_hr_input
ischemic_stress = oldpeak_input * (1 if exercise_angina_input == 1 else 0.5)

# 2. Replicate the Age Risk groups categorization
age_group_middle = 1 if (45 <= age_input < 65) else 0
age_group_senior = 1 if (age_input >= 65) else 0

# 3. Construct the dataframe to mirror the fit environment exactly
input_data = {
    'Age': age_input,
    'Cholesterol': cholesterol_input,
    'FastingBS': fasting_bs_input,
    'MaxHR': max_hr_input,
    'ExerciseAngina': exercise_angina_input,
    'Oldpeak': oldpeak_input,
    'RestingBP': resting_bp_input,
    
    # Engineered features
    'Cardiac_Workload': cardiac_workload,
    'Ischemic_Stress_Index': ischemic_stress,
    'Age_Risk_Group_Middle-Aged': age_group_middle,
    'Age_Risk_Group_Senior': age_group_senior,
    
    # One-Hot Encoded Sex (Model expects Sex_M)
    'Sex_M': 1 if sex_input == 'M' else 0,
    
    # One-Hot Encoded ST_Slope (Model expects _Flat and _Up)
    'ST_Slope_Flat': 1 if st_slope_input == 'Flat' else 0,
    'ST_Slope_Up': 1 if st_slope_input == 'Up' else 0,
    
    # Categorical variables matching exact training cases
    'ChestPainType_Ata': 1 if chest_pain == 'ATA' else 0,
    'ChestPainType_Nap': 1 if chest_pain == 'NAP' else 0,
    'ChestPainType_Ta': 1 if chest_pain == 'TA' else 0,
    
    'RestingECG_LVH': 1 if resting_ecg == 'LVH' else 0,
    'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
    'RestingECG_St': 1 if resting_ecg == 'ST' else 0  
}

# Convert dictionary to DataFrame
input_df = pd.DataFrame([input_data])

# Enforce the final strict column order seen during training
feature_order = [
    'Age', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'RestingBP',
    'Cardiac_Workload', 'Ischemic_Stress_Index', 'Age_Risk_Group_Middle-Aged', 'Age_Risk_Group_Senior',
    'Sex_M', 'ST_Slope_Flat', 'ST_Slope_Up',
    'ChestPainType_Ata', 'ChestPainType_Nap', 'ChestPainType_Ta', 
    'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_St'
]
input_df = input_df[feature_order]

# --- STEP 4: PREDICTION PIPELINE ---
st.write("### Patient Metrics Summary")
st.dataframe(input_df)

if st.button("Run Risk Assessment"):
    try:
        # BYPASS FIX: Convert input_df to a raw numpy array using .values
        # This strips the string column names so the scaler only evaluates the raw numbers
        raw_features = input_df.values
        
        # Scale the features using the raw numeric array
        scaled_input = loaded_scaler.transform(raw_features)
        
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
            
    except Exception as prediction_error:
        st.error(f"An unexpected error occurred during valuation: {prediction_error}")
