import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon=":telephone_receiver:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and icons
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header::before {
        content: "\\1F4DE";
        margin-right: 15px;
    }
    .prediction-container {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .churn-risk {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .no-churn-risk {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
    }
    .probability-text {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .high-risk::before {
        content: "\\26A0\\FE0F";
        margin-right: 10px;
    }
    .low-risk::before {
        content: "\\2705";
        margin-right: 10px;
    }
    .section-header {
        display: flex;
        align-items: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    .personal-icon::before { content: "\\1F464"; margin-right: 8px; }
    .account-icon::before { content: "\\1F4CB"; margin-right: 8px; }
    .phone-icon::before { content: "\\1F4DE"; margin-right: 8px; }
    .internet-icon::before { content: "\\1F310"; margin-right: 8px; }
    .streaming-icon::before { content: "\\1F4FA"; margin-right: 8px; }
    .billing-icon::before { content: "\\1F4B3"; margin-right: 8px; }
    .predict-icon::before { content: "\\1F52E"; margin-right: 8px; }
    .risk-factor {
        margin: 5px 0;
        padding: 10px;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 3px;
        color: #856404 !important;
        font-weight: 500;
    }
    .risk-factor::before {
        content: "\\1F538";
        margin-right: 8px;
    }
    .success-factor {
        margin: 5px 0;
        padding: 10px;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 3px;
        color: #155724 !important;
        font-weight: 500;
    }
    .success-factor::before {
        content: "\\2705";
        margin-right: 8px;
    }
    /* Dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        .risk-factor {
            background-color: #664d03;
            color: #fff3cd !important;
            border-left-color: #ffca2c;
        }
        .success-factor {
            background-color: #0f5132;
            color: #d1e7dd !important;
            border-left-color: #198754;
        }
    }
    /* Streamlit dark theme compatibility */
    .stApp[data-theme="dark"] .risk-factor {
        background-color: #664d03;
        color: #fff3cd !important;
        border-left-color: #ffca2c;
    }
    .stApp[data-theme="dark"] .success-factor {
        background-color: #0f5132;
        color: #d1e7dd !important;
        border-left-color: #198754;
    }
    /* Fix for metric labels getting cut off */
    div[data-testid="metric-container"] {
        width: 100% !important;
        min-width: 120px !important;
    }
    div[data-testid="metric-container"] > div {
        width: 100% !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }
    div[data-testid="metric-container"] label {
        font-size: 14px !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }
    .footer-heart::before {
        content: "\\2764\\FE0F";
        margin: 0 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = joblib.load("telco_churn_stacked_model.pkl")
        scaler = joblib.load("telco_scaler.pkl")
        
        # Load feature columns
        with open("telco_feature_columns.txt") as f:
            feature_columns = [line.strip() for line in f]
        
        return model, scaler, feature_columns
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, None

def preprocess_input(data, scaler):
    """
    Preprocess raw input data to match the exact preprocessing pipeline from training
    """
    df = pd.DataFrame([data])
    
    # 1. Label encode binary categorical columns
    binary_map = {'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0}
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    
    for col in binary_cols:
        df[col] = df[col].map(binary_map)
    
    # 2. One-hot encode multi-class categorical columns
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cols)
    
    # 3. Feature Engineering: Tenure binning
    conditions = [
        ((df['tenure'] >= 0) & (df['tenure'] <= 12)),
        ((df['tenure'] > 12) & (df['tenure'] <= 24)),
        ((df['tenure'] > 24) & (df['tenure'] <= 36)),
        ((df['tenure'] > 36) & (df['tenure'] <= 48)),
        ((df['tenure'] > 48) & (df['tenure'] <= 60)),
        (df['tenure'] > 60)
    ]
    choices = [0, 1, 2, 3, 4, 5]
    df['tenure_range'] = np.select(conditions, choices)
    
    # One-hot encode tenure_range
    df = pd.get_dummies(df, columns=['tenure_range'], prefix='tenure_group')
    
    # 4. Create log transformations
    df['TotalCharges_log'] = np.log1p(df['TotalCharges'])
    df['MonthlyCharges_log'] = np.log1p(df['MonthlyCharges'])
    
    # 5. Scale numerical features
    scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[scale_cols] = scaler.transform(df[scale_cols])
    
    # 6. Select only the top features used in the final model
    top_features = [
        'tenure', 'TotalCharges', 'MonthlyCharges',
        'Contract_Month-to-month', 'OnlineSecurity_No',
        'TechSupport_No', 'PaymentMethod_Electronic check',
        'Contract_Two year', 'InternetService_Fiber optic', 'gender',
        'OnlineBackup_No', 'Partner', 'DeviceProtection_No',
        'tenure_group_0', 'Dependents', 'PaperlessBilling',
        'Contract_One year', 'PaymentMethod_Bank transfer (automatic)'
    ]
    
    # Ensure all required columns exist
    for col in top_features:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the top features
    df_final = df[top_features]
    
    return df_final

def create_probability_gauge(probability):
    """Create a gauge chart for churn probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">Telco Customer Churn Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model and scaler
    model, scaler, feature_columns = load_model_and_scaler()
    
    if model is None:
        st.error("Failed to load model. Please ensure model files are available.")
        return
    
    # Sidebar for input
    st.sidebar.header("Customer Information")
    st.sidebar.markdown("Please fill in the customer details below:")
    
    # Collect input data
    with st.sidebar:
        # Personal Information
        st.markdown('<div class="section-header personal-icon">Personal Details</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        
        # Account Information
        st.markdown('<div class="section-header account-icon">Account Details</div>', unsafe_allow_html=True)
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0, step=0.1)
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0, step=0.1)
        
        # Services
        st.markdown('<div class="section-header phone-icon">Phone Services</div>', unsafe_allow_html=True)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        
        st.markdown('<div class="section-header internet-icon">Internet Services</div>', unsafe_allow_html=True)
        internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        
        st.markdown('<div class="section-header streaming-icon">Streaming Services</div>', unsafe_allow_html=True)
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        st.markdown('<div class="section-header billing-icon">Billing Information</div>', unsafe_allow_html=True)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", 
            "Mailed check", 
            "Bank transfer (automatic)", 
            "Credit card (automatic)"
        ])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Customer Profile Summary")
        
        # Display customer information in a nice format with wider columns to prevent truncation
        st.subheader("Demographics")
        demo_cols = st.columns(2)
        with demo_cols[0]:
            st.metric("Gender", gender)
            st.metric("Has Partner", partner)
        with demo_cols[1]:
            st.metric("Senior Citizen", "Yes" if senior_citizen == 1 else "No")
            st.metric("Has Dependents", dependents)
        
        st.subheader("Account")
        acc_cols = st.columns(3)
        with acc_cols[0]:
            st.metric("Tenure", f"{tenure} months")
        with acc_cols[1]:
            st.metric("Monthly Charges", f"{monthly_charges:.2f}")
        with acc_cols[2]:
            st.metric("Total Charges", f"{total_charges:.2f}")
        
        acc_cols2 = st.columns(2)
        with acc_cols2[0]:
            st.metric("Contract", contract)
        with acc_cols2[1]:
            st.metric("Payment Method", payment_method)
        
        st.subheader("Services")
        serv_cols = st.columns(2)
        with serv_cols[0]:
            st.metric("Phone Service", phone_service)
            st.metric("Internet Service", internet_service)
            st.metric("Online Security", online_security)
        with serv_cols[1]:
            st.metric("Tech Support", tech_support)
            st.metric("Streaming TV", streaming_tv)
            st.metric("Paperless Billing", paperless_billing)
    
    with col2:
        st.header("Prediction")
        
        # Predict button
        if st.button("Predict Churn Risk", type="primary", use_container_width=True):
            # Prepare input data
            input_data = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            try:
                # Preprocess and predict
                with st.spinner("Analyzing customer data..."):
                    processed_data = preprocess_input(input_data, scaler)
                    prediction = model.predict(processed_data)[0]
                    probability = model.predict_proba(processed_data)[0][1]
                
                # Display results
                prediction_label = "Churn" if int(prediction) == 1 else "No Churn"
                
                # Color-coded prediction box
                if prediction == 1:
                    st.markdown(f'''
                    <div class="prediction-container churn-risk">
                        <h3 class="high-risk" style="color: #d32f2f; text-align: center;">HIGH CHURN RISK</h3>
                        <p class="probability-text" style="color: #d32f2f;">
                            This customer is likely to churn
                        </p>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="prediction-container no-churn-risk">
                        <h3 class="low-risk" style="color: #388e3c; text-align: center;">LOW CHURN RISK</h3>
                        <p class="probability-text" style="color: #388e3c;">
                            This customer is likely to stay
                        </p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Probability gauge
                st.plotly_chart(create_probability_gauge(probability), use_container_width=True)
                
                # Detailed metrics
                st.subheader("Detailed Analysis")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Churn Probability", f"{probability:.1%}")
                with col_b:
                    st.metric("Retention Probability", f"{1-probability:.1%}")
                
                # Risk factors (simplified analysis)
                st.subheader("Key Risk Factors")
                risk_factors = []
                
                if contract == "Month-to-month":
                    risk_factors.append("Month-to-month contract increases churn risk")
                if payment_method == "Electronic check":
                    risk_factors.append("Electronic check payment method")
                if online_security == "No":
                    risk_factors.append("No online security service")
                if tech_support == "No":
                    risk_factors.append("No tech support service")
                if tenure < 12:
                    risk_factors.append("New customer (tenure < 1 year)")
                if monthly_charges > 70:
                    risk_factors.append("High monthly charges")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f'<div class="risk-factor">{factor}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-factor">No major risk factors identified</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div>Built by Disha using Streamlit | '
        'Model: Stacked Ensemble (Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, MLP)</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
