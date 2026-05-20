import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. GLOBAL DASHBOARD CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Heart Failure Predictive Analytics Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. DATASET & ASSET UTILITIES
# ==========================================
def get_db_data(query):
    """Establishes a connection to the SQLite database and returns a DataFrame."""
    try:
        # Connects to your project's SQLite database file
        conn = sqlite3.connect('heart_failure_project.db')
        data = pd.read_sql_query(query, conn)
        conn.close()
        return data
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None

# ==========================================
# 3. SIDEBAR NAVIGATION CONTROLLER
# ==========================================
st.sidebar.image("https://img.icons8.com/illustrations/opacity/100/heart-health.png", width=100)
st.sidebar.title("TS Academy Capstone")
st.sidebar.markdown("**Group:** NN  \n**Track:** Classification Depth Track")
st.sidebar.divider()

# Centralized Navigation Router
page_selection = st.sidebar.radio(
    "Dashboard Navigation Pages",
    options=[
        "1. Project Overview & EDA", 
        "2. Interactive SQL Explorer", 
        "3. Unsupervised Personas",
        "4. Patient Risk Assessment Tool"
    ]
)

# ==========================================
# PAGE 1: PROJECT OVERVIEW & EDA
# ==========================================
if page_selection == "1. Project Overview & EDA":
    st.title("❤️ Heart Failure Patient Risk Analytics")
    st.markdown("""
    This dashboard serves as the operational interface for our predictive pipeline, stratifying patient risk profiles 
    to prevent acute cardiac events through data-driven clinical screening.
    """)
    st.divider()
    
    # Try to load overview records from the database
    df_display = get_db_data("SELECT * FROM heart_data LIMIT 100")
    
    if df_display is not None:
        # Layout metrics side-by-side
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Analyzed Cohort", value="918 Patients")
        with col2:
            st.metric(label="Baseline Target Disease Rate", value="55.3%")
        with col3:
            st.metric(label="Data Retention Post-Cleaning", value="81.1%")
            
        st.subheader("📋 Core Clinical Matrix Sample (First 5 Rows)")
        st.dataframe(df_display.head(5), use_container_width=True)
    else:
        st.info("ℹ️ Overview metrics ready. Ensure 'heart_failure_project.db' is populated to visualize live row data.")

# ==========================================
# PAGE 2: INTERACTIVE SQL EXPLORER
# ==========================================
elif page_selection == "2. Interactive SQL Explorer":
    st.title("🔍 Multi-Criteria SQL Data Explorer")
    st.markdown("Query the active database infrastructure in real-time to isolate specific demographic risk signals.")
    st.divider()
    
    # Pre-set Dropdown Selection for Grading Prompts
    query_preset = st.selectbox(
        "Select a Pre-Configured Clinical Analysis Query:",
        options=[
            "Custom User Query",
            "Risk Stratification by Chest Pain Type",
            "Physiological Strain by Age Cohort",
            "Data Integrity Audit (Zero Counts)"
        ]
    )
    
    # Map selection to query strings
    if query_preset == "Risk Stratification by Chest Pain Type":
        sql_input = "SELECT ChestPainType, COUNT(*) as total, SUM(HeartDisease) as positive_cases, ROUND(AVG(HeartDisease)*100, 2) as risk_pct FROM heart_data GROUP BY ChestPainType ORDER BY risk_pct DESC;"
    elif query_preset == "Physiological Strain by Age Cohort":
        sql_input = "SELECT CASE WHEN Age < 40 THEN 'Under 40' WHEN Age BETWEEN 40 AND 60 THEN '40-60' ELSE 'Over 60' END as age_group, ExerciseAngina, COUNT(*), ROUND(AVG(MaxHR), 1) as avg_max_hr FROM heart_data GROUP BY age_group, ExerciseAngina;"
    elif query_preset == "Data Integrity Audit (Zero Counts)":
        sql_input = "SELECT HeartDisease, COUNT(*) as total, SUM(CASE WHEN Cholesterol = 0 THEN 1 ELSE 0 END) as zero_chol_count FROM heart_data GROUP BY HeartDisease;"
    else:
        sql_input = st.text_area("Write Your Custom SQL Query Here:", value="SELECT * FROM heart_data WHERE Age > 60 LIMIT 10;")
        
    st.code(sql_input, language="sql")
    
    if st.button("Execute Database Query", type="primary"):
        query_result = get_db_data(sql_input)
        if query_result is not None:
            st.success("Query Executed Successfully!")
            st.dataframe(query_result, use_container_width=True)

# ==========================================
# PAGE 3: UNSUPERVISED PERSONAS
# ==========================================
elif page_selection == "3. Unsupervised Personas":
    st.title("🧬 Unsupervised Patient Stratification Profiles")
    st.markdown("Review the distinct mathematical patient cohorts generated by our K-Means clustering pipeline.")
    st.divider()
    
    # Dynamic Selector for Cluster Focus
    selected_cluster = st.selectbox(
        "Select a Clinical Persona Profile to Inspect:",
        options=[
            "Cluster 0: Low-Risk Vitality Patients", 
            "Cluster 1: Asymptomatic Metabolic Risk Cohort", 
            "Cluster 2: Acute Ischemic / Critical Risk Patients"
        ]
    )
    
    # Conditional logic displaying insights per selection
    if "Cluster 0" in selected_cluster:
        st.info("""
        💡 **Cluster 0: Low-Risk Vitality Patients**
        * **Profile:** Lower average age, healthy high maximum heart rates, minimal ST-depression markers.
        * **Clinical Target Strategy:** Optimal candidates for lower insurance premium tiers and standard preventative monitoring.
        """)
    elif "Cluster 1" in selected_cluster:
        st.warning("""
        ⚠️ **Cluster 1: Asymptomatic Metabolic Risk Cohort**
        * **Profile:** Middle-aged to older individuals presenting with elevated resting blood pressures and high cholesterol readings but low stress-test strain.
        * **Clinical Target Strategy:** High-priority outpatient diagnostic outreach; preventative lifestyle and metabolic coaching interventions.
        """)
    elif "Cluster 2" in selected_cluster:
        st.error("""
        🚨 **Cluster 2: Acute Ischemic / Critical Risk Patients**
        * **Profile:** Older individuals displaying severely elevated ST-depression metrics and low peak heart rate thresholds. High heart disease saturation.
        * **Clinical Target Strategy:** Immediate cardiological tracking, priority clinical resource allocation, and intensive care management assignment.
        """)

# ==========================================
# PAGE 4: PATIENT RISK ASSESSMENT TOOL
# ==========================================
elif page_selection == "4. Patient Risk Assessment Tool":
    st.title("🩺 Live Patient Risk Prediction Interface")
    st.markdown("Input raw clinical biometric markers to evaluate a patient's immediate probability of heart disease using our deployed model framework.")
    st.divider()

    try:
        # Load the saved model and scaler files
        with open('heart_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            loaded_scaler = pickle.load(f)
            
        # Create a clean input form split across two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Continuous Vital Signs")
            age = st.slider("Patient Age", min_value=1, max_value=100, value=50)
            resting_bp = st.slider("Resting Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
            cholesterol = st.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=500, value=200)
            max_hr = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=210, value=150)
            oldpeak = st.slider("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)

        with col2:
            st.subheader("🫀 Categorical Clinical Observations")
            sex = st.selectbox("Biological Sex", options=["Male", "Female"])
            chest_pain = st.selectbox("Chest Pain Type", options=["Asymptomatic (ASY)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Typical Angina (TA)"])
            resting_ecg = st.selectbox("Resting ECG Results", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy (LVH)"])
            exercise_angina = st.radio("Exercise-Induced Angina?", options=["Yes", "No"])
            st_slope = st.selectbox("Peak Exercise ST Slope", options=["Up-sloping", "Flat", "Down-sloping"])

        # --- FEATURE ENGINEERING PIPELINE ---
        # Match Part 2 feature calculations exactly
        cardiac_workload = resting_bp / (max_hr + 1)
        angina_binary = 1 if exercise_angina == "Yes" else 0
        ischemic_stress = oldpeak * angina_binary

        # Build raw feature alignment record
        input_data = {
            'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol, 
            'MaxHR': max_hr, 'Oldpeak': oldpeak, 'ExerciseAngina': angina_binary,
            'Cardiac_Workload': cardiac_workload, 'Ischemic_Stress_Index': ischemic_stress,
            'Sex_M': 1 if sex == "Male" else 0,
            'ChestPainType_ATA': 1 if "ATA" in chest_pain else 0,
            'ChestPainType_NAP': 1 if "NAP" in chest_pain else 0,
            'ChestPainType_TA': 1 if "TA" in chest_pain else 0,
            'RestingECG_ST': 1 if "ST" in resting_ecg else 0,
            'RestingECG_LVH': 1 if "LVH" in resting_ecg else 0,
            'ST_Slope_Flat': 1 if "Flat" in st_slope else 0,
            'ST_Slope_Up': 1 if "Up" in st_slope else 0,
            'Age_Risk_Group_Middle-Aged': 1 if 45 < age <= 60 else 0,
            'Age_Risk_Group_Senior': 1 if age > 60 else 0
        }

        # Format input vector into DataFrame matching the structure of X_encoded
        input_df = pd.DataFrame([input_data])
        
        # Scale inputs using the saved training scaling rules
        scaled_input = loaded_scaler.transform(input_df)

        # --- EXECUTE PREDICTION ---
        st.divider()
        if st.button("Run Diagnostic Risk Assessment", type="primary"):
            # Predict probability of positive case (HeartDisease = 1)
            risk_probability = loaded_model.predict_proba(scaled_input)[0][1]
            
            st.subheader("Diagnostic Evaluation Result")
            # Apply your business cost threshold optimized in Part 5
            if risk_probability >= 0.40: 
                st.error(f"🚨 **High Risk Warning:** This patient has a **{risk_probability*100:.1f}%** probability of underlying heart disease. Prioritize for secondary clinical screening.")
            else:
                st.success(f"✅ **Low Risk Profile:** This patient has a **{risk_probability*100:.1f}%** probability of heart disease. Standard preventive tracking recommended.")

    except FileNotFoundError:
        st.warning("⚠️ **Model Assets Missing:** Please ensure 'heart_model.pkl' and 'scaler.pkl' are downloaded from your notebook and saved in this directory.")
