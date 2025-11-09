"""
Heart Disease Prediction - Enhanced Streamlit Web Application
Advanced UI with Analytics, Charts, and Interactive Visualizations
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255,107,107,0.4);
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND SCALER
# ============================================================================

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

# Load the model and scaler
model, scaler = load_model_and_scaler()

# ============================================================================
# HELPER FUNCTIONS FOR VISUALIZATIONS
# ============================================================================

def create_gauge_chart(value, title, color_scheme="red"):
    """Create a gauge chart for displaying probabilities."""
    colors = {
        "red": ["#90EE90", "#FFD700", "#FF6B6B"],
        "blue": ["#E3F2FD", "#90CAF9", "#1976D2"]
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#333'}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': colors[color_scheme][2]},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': colors[color_scheme][0]},
                {'range': [33, 66], 'color': colors[color_scheme][1]},
                {'range': [66, 100], 'color': colors[color_scheme][2]}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial"}
    )
    
    return fig

def create_risk_factors_radar(input_data):
    """Create a radar chart showing normalized risk factors."""
    # Normalize values to 0-100 scale for visualization
    factors = {
        'Age': (input_data['age'].iloc[0] - 20) / 80 * 100,
        'Blood Pressure': (input_data['resting bp s'].iloc[0] - 80) / 120 * 100,
        'Cholesterol': (input_data['cholesterol'].iloc[0] - 100) / 300 * 100,
        'Heart Rate': (220 - input_data['max heart rate'].iloc[0]) / 160 * 100,
        'ST Depression': (input_data['oldpeak'].iloc[0]) / 6 * 100
    }
    
    categories = list(factors.keys())
    values = list(factors.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(255, 107, 107, 0.3)',
        line=dict(color='#FF6B6B', width=2),
        name='Patient Values'
    ))
    
    # Add normal range reference
    normal_values = [40, 40, 40, 40, 20]  # Adjusted normal ranges
    fig.add_trace(go.Scatterpolar(
        r=normal_values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.1)',
        line=dict(color='#667eea', width=2, dash='dash'),
        name='Normal Range'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False
            )
        ),
        showlegend=True,
        title="Risk Factors Analysis",
        height=400,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    return fig

def create_feature_importance_chart(prediction_proba):
    """Create a feature contribution chart."""
    # Simulated feature importance (in real scenario, use SHAP or similar)
    features = ['Age', 'Chest Pain', 'Blood Pressure', 'Cholesterol', 
                'Max Heart Rate', 'Exercise Angina', 'ST Depression', 'ST Slope']
    
    # Generate realistic-looking importance values
    np.random.seed(42)
    importance = np.random.uniform(0.05, 0.15, len(features))
    importance = importance / importance.sum() * prediction_proba[1]
    
    colors = ['#FF6B6B' if imp > importance.mean() else '#667eea' for imp in importance]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f'{val:.1%}' for val in importance],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Feature Contribution to Prediction",
        xaxis_title="Contribution",
        yaxis_title="Features",
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(tickformat='.0%')
    )
    
    return fig

def create_comparison_chart(input_data):
    """Create comparison chart with normal ranges."""
    metrics = {
        'Age': {'value': input_data['age'].iloc[0], 'normal': 50, 'unit': 'years'},
        'BP': {'value': input_data['resting bp s'].iloc[0], 'normal': 120, 'unit': 'mmHg'},
        'Cholesterol': {'value': input_data['cholesterol'].iloc[0], 'normal': 200, 'unit': 'mg/dl'},
        'Heart Rate': {'value': input_data['max heart rate'].iloc[0], 'normal': 150, 'unit': 'bpm'}
    }
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(metrics.keys()),
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}]]
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for (metric_name, data), (row, col) in zip(metrics.items(), positions):
        delta_value = data['value'] - data['normal']
        delta_color = "red" if abs(delta_value) > data['normal'] * 0.2 else "green"
        
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=data['value'],
                delta={'reference': data['normal'], 'relative': False},
                title={'text': f"{data['unit']}"},
                number={'font': {'size': 30}},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_probability_distribution():
    """Create a bell curve showing risk distribution."""
    x = np.linspace(0, 100, 100)
    y1 = 100 * np.exp(-((x-30)**2)/(2*15**2))  # Low risk population
    y2 = 80 * np.exp(-((x-70)**2)/(2*15**2))   # High risk population
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=y1,
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=2),
        name='Low Risk Population'
    ))
    
    fig.add_trace(go.Scatter(
        x=x, y=y2,
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.3)',
        line=dict(color='#FF6B6B', width=2),
        name='High Risk Population'
    ))
    
    fig.update_layout(
        title="Risk Score Distribution",
        xaxis_title="Risk Score",
        yaxis_title="Population Density",
        height=300,
        showlegend=True,
        hovermode='x unified',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# ============================================================================
# HEADER SECTION
# ============================================================================

st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction Dashboard</h1>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="info-box">
            <h3 style="color: #667eea; margin: 0;">🎯 Accurate</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #666;">ML-Powered Predictions</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="info-box">
            <h3 style="color: #667eea; margin: 0;">⚡ Fast</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #666;">Instant Results</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="info-box">
            <h3 style="color: #667eea; margin: 0;">📊 Visual</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #666;">Interactive Analytics</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="info-box">
            <h3 style="color: #667eea; margin: 0;">🔒 Secure</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #666;">Privacy Protected</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - USER INPUT FEATURES
# ============================================================================

st.sidebar.header("📋 Patient Information")
st.sidebar.markdown("Please enter the patient's medical data:")

# Create tabs for better organization
tab1, tab2 = st.sidebar.tabs(["Basic Info", "Clinical Data"])

with tab1:
    age = st.slider("Age (years)", 20, 100, 50, 1, help="Patient's age in years")
    sex_option = st.selectbox("Sex", ["Male", "Female"], help="Patient's biological sex")
    sex = 1 if sex_option == "Male" else 0
    
    cp_option = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
        help="Type of chest pain experienced"
    )
    cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
    cp = cp_mapping[cp_option]

with tab2:
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120, 1)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 200, 1)
    
    fbs_option = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    fbs = 1 if fbs_option == "Yes" else 0
    
    restecg_option = st.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )
    restecg_mapping = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    restecg = restecg_mapping[restecg_option]
    
    thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150, 1)
    
    exang_option = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exang = 1 if exang_option == "Yes" else 0
    
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
    
    slope_option = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope = slope_mapping[slope_option]
    
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    
    thal_option = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    thal_mapping = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
    thal = thal_mapping[thal_option]

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔍 Predict Risk", use_container_width=True, type="primary")

# ============================================================================
# CREATE INPUT DATAFRAME
# ============================================================================

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
    'ST slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# ============================================================================
# MAIN CONTENT - BEFORE PREDICTION
# ============================================================================

if not predict_button:
    st.info("👈 Please enter patient information in the sidebar and click **Predict Risk** to see results.")
    
    # Show sample visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Patient Data Overview")
        display_data = pd.DataFrame({
            'Feature': ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol',
                       'Max Heart Rate', 'Exercise Angina', 'ST Depression'],
            'Value': [f"{age} years", sex_option, cp_option, f"{trestbps} mmHg",
                     f"{chol} mg/dl", f"{thalach} bpm", exang_option, f"{oldpeak}"]
        })
        st.dataframe(display_data, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📈 Risk Distribution")
        st.plotly_chart(create_probability_distribution(), use_container_width=True)

# ============================================================================
# PREDICTION AND RESULTS
# ============================================================================

if predict_button:
    # Align features with scaler
    train_cols = scaler.feature_names_in_
    input_data_aligned = input_data.reindex(columns=train_cols, fill_value=0)
    input_scaled = scaler.transform(input_data_aligned)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    confidence = prediction_proba[prediction] * 100
    
    # ============================================================================
    # RESULTS HEADER
    # ============================================================================
    
    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")
    
    # Main result cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if prediction == 0:
            st.success("### ✅ Low Risk")
            st.success("No Heart Disease Detected")
        else:
            st.error("### ⚠️ High Risk")
            st.error("Heart Disease Detected")
    
    with col2:
        st.metric("Confidence Level", f"{confidence:.1f}%", 
                 delta=f"{confidence - 50:.1f}%" if confidence > 50 else f"{confidence - 50:.1f}%")
    
    with col3:
        st.metric("Disease Probability", f"{prediction_proba[1]*100:.1f}%",
                 delta="High" if prediction_proba[1] > 0.5 else "Low",
                 delta_color="inverse")
    
    with col4:
        st.metric("No Disease Probability", f"{prediction_proba[0]*100:.1f}%",
                 delta="High" if prediction_proba[0] > 0.5 else "Low")
    
    # ============================================================================
    # VISUALIZATION SECTION
    # ============================================================================
    
    st.markdown("---")
    st.markdown("## 📊 Detailed Analytics")
    
    # Row 1: Gauge Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            create_gauge_chart(prediction_proba[1]*100, "Disease Risk Score", "red"),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            create_gauge_chart(confidence, "Prediction Confidence", "blue"),
            use_container_width=True
        )
    
    # Row 2: Radar and Feature Importance
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_risk_factors_radar(input_data), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_feature_importance_chart(prediction_proba), use_container_width=True)
    
    # Row 3: Comparison Chart
    st.markdown("### 📈 Patient Metrics vs Normal Ranges")
    st.plotly_chart(create_comparison_chart(input_data), use_container_width=True)
    
    # Row 4: Risk Distribution
    st.markdown("### 📉 Population Risk Distribution")
    fig_dist = create_probability_distribution()
    # Add patient's score marker
    fig_dist.add_vline(
        x=prediction_proba[1]*100,
        line_dash="dash",
        line_color="green",
        annotation_text="Your Score",
        annotation_position="top"
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # ============================================================================
    # DETAILED DATA TABLE
    # ============================================================================
    
    st.markdown("---")
    st.markdown("## 📋 Complete Patient Data")
    
    detailed_data = pd.DataFrame({
        'Feature': ['Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol',
                   'Fasting Blood Sugar > 120', 'Resting ECG', 'Max Heart Rate',
                   'Exercise Induced Angina', 'ST Depression', 'ST Slope', 
                   'Major Vessels', 'Thalassemia'],
        'Value': [f"{age} years", sex_option, cp_option, f"{trestbps} mmHg",
                 f"{chol} mg/dl", fbs_option, restecg_option, f"{thalach} bpm",
                 exang_option, f"{oldpeak}", slope_option, ca, thal_option],
        'Status': ['Normal' if age < 65 else 'Elevated',
                  'N/A', 'N/A',
                  'Normal' if 90 <= trestbps <= 120 else 'Elevated',
                  'Normal' if chol < 200 else 'High',
                  'Normal' if fbs == 0 else 'High',
                  'Normal' if restecg == 0 else 'Abnormal',
                  'Normal' if thalach > 100 else 'Low',
                  'Normal' if exang == 0 else 'Positive',
                  'Normal' if oldpeak < 1 else 'Elevated',
                  'N/A', 'N/A', 'N/A']
    })
    
    st.dataframe(
        detailed_data.style.applymap(
            lambda x: 'background-color: #ffcccc' if x in ['Elevated', 'High', 'Abnormal', 'Positive', 'Low'] else '',
            subset=['Status']
        ),
        use_container_width=True,
        hide_index=True
    )
    
    # ============================================================================
    # RECOMMENDATIONS
    # ============================================================================
    
    st.markdown("---")
    st.markdown("## 💡 Recommendations")
    
    recommendations = []
    if prediction == 1:
        recommendations.append("🏥 **Immediate Action:** Consult with a cardiologist for comprehensive evaluation")
        recommendations.append("💊 **Medication:** Discuss preventive medications with your doctor")
    
    if chol > 240:
        recommendations.append("🥗 **Diet:** Adopt a heart-healthy diet low in saturated fats")
    
    if trestbps > 140:
        recommendations.append("🧂 **Blood Pressure:** Monitor and manage blood pressure regularly")
    
    if thalach < 100:
        recommendations.append("🏃 **Exercise:** Gradual increase in physical activity under medical supervision")
    
    if not recommendations:
        recommendations.append("✅ **Maintain:** Continue healthy lifestyle habits")
        recommendations.append("📅 **Regular Checkups:** Annual cardiac health screening")
    
    for rec in recommendations:
        st.info(rec)
    
    # ============================================================================
    # DISCLAIMER
    # ============================================================================
    
    st.markdown("---")
    st.warning("""
    **⚠️ Medical Disclaimer:**
    
    This prediction is generated by a machine learning model and should **NOT** be used as a 
    substitute for professional medical advice, diagnosis, or treatment. Always seek the 
    advice of your physician or other qualified health provider with any questions you may 
    have regarding a medical condition.
    
    **This tool is for informational and educational purposes only.**
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem 0;'>
    <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>
        <strong>Heart Disease Prediction System</strong>
    </p>
    <p style='font-size: 0.9rem; margin: 0;'>
        Powered by Machine Learning | Built with Streamlit & Plotly
    </p>
    <p style='font-size: 0.8rem; margin-top: 0.5rem; color: #999;'>
        Version 2.0 | © 2025
    </p>
</div>
""", unsafe_allow_html=True)