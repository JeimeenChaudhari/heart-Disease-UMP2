import streamlit as st
import pandas as pd
import pickle
import numpy as np

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOADING MODEL AND SCALER

@st.cache_resource
def load_model_and_scaler():
    """Load the pre-trained model and scaler from pickle files."""
    try:
        with open('heart_disease_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"⚠️ Required file not found: {str(e)}")
        st.error("Please run 'train_model.py' first to generate all required files.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.stop()

# Loading the model and scaler
model, scaler = load_model_and_scaler()

# HEADER SECTION

st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("""
This application uses a Machine Learning model to predict the risk of heart disease 
based on patient medical data. Please enter the patient information in the sidebar 
and click **Predict** to see the results.
""")

st.markdown("---")

# SIDEBAR - USER INPUT FEATURES
st.sidebar.header("📋 Patient Information")
st.sidebar.markdown("Please enter the patient's medical data:")

# Age
age = st.sidebar.slider(
    "Age (years)",
    min_value=20,
    max_value=100,
    value=50,
    step=1,
    help="Patient's age in years"
)

# Sex
sex_option = st.sidebar.selectbox(
    "Sex",
    options=["Male", "Female"],
    help="Patient's biological sex"
)
sex = 1 if sex_option == "Male" else 0

# Chest Pain Type (cp)
cp_option = st.sidebar.selectbox(
    "Chest Pain Type",
    options=[
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ],
    help="Type of chest pain experienced"
)
cp_mapping = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
cp = cp_mapping[cp_option]

# Resting Blood Pressure (trestbps)
trestbps = st.sidebar.slider(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=200,
    value=120,
    step=1,
    help="Resting blood pressure in mm Hg"
)

# Serum Cholesterol (chol)
chol = st.sidebar.slider(
    "Serum Cholesterol (mg/dl)",
    min_value=100,
    max_value=400,
    value=200,
    step=1,
    help="Serum cholesterol level in mg/dl"
)

# Fasting Blood Sugar (fbs)
fbs_option = st.sidebar.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    options=["No", "Yes"],
    help="Is fasting blood sugar greater than 120 mg/dl?"
)
fbs = 1 if fbs_option == "Yes" else 0

# Resting ECG (restecg)
restecg_option = st.sidebar.selectbox(
    "Resting ECG Results",
    options=[
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ],
    help="Resting electrocardiographic results"
)
restecg_mapping = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
restecg = restecg_mapping[restecg_option]

# Maximum Heart Rate (thalach)
thalach = st.sidebar.slider(
    "Maximum Heart Rate Achieved",
    min_value=60,
    max_value=220,
    value=150,
    step=1,
    help="Maximum heart rate achieved during exercise"
)

# Exercise Induced Angina (exang)
exang_option = st.sidebar.selectbox(
    "Exercise Induced Angina",
    options=["No", "Yes"],
    help="Does exercise induce angina?"
)
exang = 1 if exang_option == "Yes" else 0

# ST Depression (oldpeak)
oldpeak = st.sidebar.slider(
    "ST Depression",
    min_value=0.0,
    max_value=6.0,
    value=1.0,
    step=0.1,
    help="ST depression induced by exercise relative to rest"
)

# Slope of Peak Exercise ST Segment (slope)
slope_option = st.sidebar.selectbox(
    "Slope of Peak Exercise ST Segment",
    options=[
        "Upsloping",
        "Flat",
        "Downsloping"
    ],
    help="The slope of the peak exercise ST segment"
)
slope_mapping = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}
slope = slope_mapping[slope_option]

# PREDICTION SECTION
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔍 Predict", use_container_width=True, type="primary")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Patient Data Summary")
    
    # Create a DataFrame with user input using the exact column names from the dataset
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'chest pain type': [cp],
        'resting bp s': [trestbps],
        'cholesterol': [chol],
        'fasting blood sugar': [fbs],
        'resting ecg': [restecg],
        'max heart rate': [thalach],
        'exercise angina': [exang],
        'oldpeak': [oldpeak],
        'ST slope': [slope]
    })
    
    # Display input data in a readable format
    display_data = {
        'Feature': [
            'Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol',
            'Fasting Blood Sugar > 120', 'Resting ECG', 'Max Heart Rate',
            'Exercise Induced Angina', 'ST Depression', 'ST Slope'
        ],
        'Value': [
            f"{age} years",
            sex_option,
            cp_option,
            f"{trestbps} mm Hg",
            f"{chol} mg/dl",
            fbs_option,
            restecg_option,
            f"{thalach} bpm",
            exang_option,
            f"{oldpeak}",
            slope_option
        ]
    }
    
    st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)

with col2:
    st.subheader("ℹ️ Information")
    st.info("""
    **How to use:**
    1. Enter patient data in the sidebar
    2. Click the **Predict** button
    3. View the prediction results
    
    **Note:** This tool is for informational purposes only and should not replace professional medical advice.
    """)

# MAKE PREDICTION
if predict_button:
    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Get confidence (probability of the predicted class)
    confidence = prediction_proba[prediction] * 100
    
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    
    # Display prediction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 0:
            st.success("### ✅ Low Risk")
            st.success("**No Heart Disease Detected**")
        else:
            st.error("### ⚠️ High Risk")
            st.error("**Heart Disease Detected**")
    
    with col2:
        st.metric(
            label="Confidence Level",
            value=f"{confidence:.2f}%"
        )
    
    with col3:
        st.metric(
            label="Risk Probability",
            value=f"{prediction_proba[1]*100:.2f}%",
            help="Probability of having heart disease"
        )
    
    # Detailed probability breakdown
    st.markdown("---")
    st.subheader("📈 Probability Breakdown")
    
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        st.metric(
            label="No Disease Probability",
            value=f"{prediction_proba[0]*100:.2f}%"
        )
    
    with prob_col2:
        st.metric(
            label="Disease Probability",
            value=f"{prediction_proba[1]*100:.2f}%"
        )
    
    # Additional information and disclaimer
    st.markdown("---")
    st.warning("""
    **⚠️ Medical Disclaimer:**
    This prediction is generated by a machine learning model and should NOT be used as a 
    substitute for professional medical advice, diagnosis, or treatment. Always seek the 
    advice of your physician or other qualified health provider with any questions you may 
    have regarding a medical condition.
    """)

else:
    # Show placeholder when no prediction has been made
    st.info(" Please enter patient information in the sidebar and click **Predict** to see results.")

# FOOTER

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Heart Disease Prediction System | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)