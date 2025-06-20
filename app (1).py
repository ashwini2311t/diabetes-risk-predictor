import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Try importing sklearn with error handling
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

@st.cache_data
def load_data():
    """Generate synthetic diabetes data"""
    if not SKLEARN_AVAILABLE:
        st.error("Required machine learning libraries are not available. Please check deployment configuration.")
        return None
        
    np.random.seed(42)
    n = 1000
    
    data = pd.DataFrame({
        'Pregnancies': np.random.poisson(3, n),
        'Glucose': np.random.normal(120, 30, n),
        'BloodPressure': np.random.normal(70, 12, n),
        'SkinThickness': np.random.normal(25, 8, n),
        'Insulin': np.random.exponential(100, n),
        'BMI': np.random.normal(28, 6, n),
        'DiabetesPedigreeFunction': np.random.exponential(0.5, n),
        'Age': np.random.normal(35, 12, n)
    })
    
    # Create realistic outcome
    risk = (
        (data['Glucose'] - 100) * 0.02 +
        (data['BMI'] - 25) * 0.05 +
        (data['Age'] - 30) * 0.01 +
        data['DiabetesPedigreeFunction'] * 0.3 +
        np.random.normal(0, 0.5, n)
    )
    data['Outcome'] = (risk > 0.5).astype(int)
    return data

@st.cache_resource
def train_models():
    """Train all models"""
    if not SKLEARN_AVAILABLE:
        return None, None, None
        
    data = load_data()
    if data is None:
        return None, None, None
    
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = data[features]
    y = data['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    trained = {}
    for name, model in models.items():
        try:
            if name in ['Logistic Regression', 'SVM']:
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)
            trained[name] = model
        except Exception as e:
            st.error(f"Error training {name}: {str(e)}")
    
    return trained, scaler, features

def get_risk_level(probability):
    """Determine risk level"""
    if probability < 0.3:
        return "Low Risk", "green"
    elif probability < 0.7:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"

def main():
    st.title("🏥 Diabetes Risk Prediction System")
    st.markdown("AI-powered diabetes risk assessment using multiple machine learning models")
    
    st.warning("⚠️ Medical Disclaimer: This tool is for educational purposes only. Consult healthcare providers for medical decisions.")
    
    if not SKLEARN_AVAILABLE:
        st.error("Machine learning libraries are not properly installed. Please contact support.")
        st.info("Required packages: scikit-learn, pandas, numpy, plotly")
        return
    
    models, scaler, features = train_models()
    
    if models is None:
        st.error("Unable to initialize machine learning models.")
        return
    
    st.sidebar.header("Patient Information")
    
    inputs = {
        'Pregnancies': st.sidebar.number_input("Number of Pregnancies", 0, 17, 1),
        'Glucose': st.sidebar.number_input("Glucose Level (mg/dL)", 0, 200, 120),
        'BloodPressure': st.sidebar.number_input("Blood Pressure (mmHg)", 0, 122, 70),
        'SkinThickness': st.sidebar.number_input("Skin Thickness (mm)", 0, 99, 25),
        'Insulin': st.sidebar.number_input("Insulin Level (μU/mL)", 0, 846, 80),
        'BMI': st.sidebar.number_input("BMI", 0.0, 67.1, 25.0, step=0.1),
        'DiabetesPedigreeFunction': st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01),
        'Age': st.sidebar.number_input("Age", 21, 81, 30)
    }
    
    if st.sidebar.button("Predict Diabetes Risk", type="primary"):
        input_df = pd.DataFrame([inputs])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Model Predictions")
            predictions = {}
            
            for name, model in models.items():
                try:
                    if name in ['Logistic Regression', 'SVM']:
                        input_scaled = scaler.transform(input_df[features])
                        prob = model.predict_proba(input_scaled)[0][1]
                    else:
                        prob = model.predict_proba(input_df[features])[0][1]
                    
                    predictions[name] = prob
                    risk_level, color = get_risk_level(prob)
                    
                    st.metric(
                        label=name,
                        value=f"{prob:.1%}",
                        delta=risk_level
                    )
                except Exception as e:
                    st.error(f"Error with {name}: {str(e)}")
        
        with col2:
            st.subheader("📊 Risk Visualization")
            
            if predictions:
                fig = go.Figure()
                model_names = list(predictions.keys())
                probs = list(predictions.values())
                colors = ['green' if p < 0.3 else 'orange' if p < 0.7 else 'red' for p in probs]
                
                fig.add_trace(go.Bar(
                    x=model_names,
                    y=probs,
                    marker_color=colors,
                    text=[f"{p:.1%}" for p in probs],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Diabetes Risk by Model",
                    xaxis_title="ML Models",
                    yaxis_title="Risk Probability",
                    yaxis=dict(range=[0, 1]),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        if predictions:
            avg_prob = np.mean(list(predictions.values()))
            risk_level, color = get_risk_level(avg_prob)
            
            st.subheader("🎯 Overall Assessment")
            st.markdown(f"**Average Risk Score:** {avg_prob:.1%}")
            st.markdown(f"**Risk Level:** <span style='color: {color}; font-weight: bold;'>{risk_level}</span>", unsafe_allow_html=True)
            
            st.subheader("💡 Health Recommendations")
            
            recommendations = []
            if avg_prob > 0.5:
                recommendations.extend([
                    "🚨 Consult with a healthcare provider immediately",
                    "📊 Consider getting HbA1c and fasting glucose tests"
                ])
            
            if inputs['BMI'] > 25:
                recommendations.append("🏃‍♂️ Focus on weight management through diet and exercise")
            
            if inputs['Glucose'] > 140:
                recommendations.append("🍎 Monitor blood sugar levels and follow a low-sugar diet")
            
            if inputs['Age'] > 45:
                recommendations.append("🔄 Regular health screenings are important at your age")
            
            recommendations.extend([
                "💪 Maintain regular physical activity (150 min/week)",
                "🥗 Follow a balanced, Mediterranean-style diet"
            ])
            
            for rec in recommendations:
                st.markdown(f"• {rec}")

# About section
def show_about():
    st.title("ℹ️ About This System")
    
    st.markdown("""
    ## AI-Powered Diabetes Risk Assessment
    
    This system uses multiple machine learning algorithms to assess diabetes risk based on key health parameters.
    
    ### 🤖 Machine Learning Models
    - **Logistic Regression**: Linear statistical approach
    - **Random Forest**: Ensemble tree-based method
    - **Support Vector Machine**: Advanced classification algorithm
    - **K-Nearest Neighbors**: Instance-based learning
    
    ### 📊 Input Parameters
    - **Pregnancies**: Number of pregnancies
    - **Glucose**: Blood glucose level (mg/dL)
    - **Blood Pressure**: Diastolic blood pressure (mmHg)
    - **Skin Thickness**: Triceps skin fold thickness (mm)
    - **Insulin**: Serum insulin level (μU/mL)
    - **BMI**: Body Mass Index (kg/m²)
    - **Diabetes Pedigree**: Diabetes pedigree function
    - **Age**: Age in years
    
    ### ⚕️ Medical Disclaimer
    This tool is designed for educational and research purposes only. It should never be used as a substitute for professional medical advice, diagnosis, or treatment.
    """)

# Navigation
page = st.sidebar.selectbox("Navigate", ["Risk Prediction", "About"])

if page == "Risk Prediction":
    main()
else:
    show_about()

st.markdown("---")
st.markdown("**Diabetes Risk Prediction System** | Educational Use Only")