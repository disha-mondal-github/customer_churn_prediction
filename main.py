from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Telco Customer Churn Predictor")

# Load your trained objects
model = joblib.load("telco_churn_stacked_model.pkl")
scaler = joblib.load("telco_scaler.pkl")

# Expected feature columns after preprocessing (from feature list)
with open("telco_feature_columns.txt") as f:
    feature_columns = [line.strip() for line in f]

# The top 18 features used in final model
top_features = [
    'tenure', 'TotalCharges', 'MonthlyCharges',
    'Contract_Month-to-month', 'OnlineSecurity_No',
    'TechSupport_No', 'PaymentMethod_Electronic check',
    'Contract_Two year', 'InternetService_Fiber optic', 'gender',
    'OnlineBackup_No', 'Partner', 'DeviceProtection_No',
    'tenure_group_0', 'Dependents', 'PaperlessBilling',
    'Contract_One year', 'PaymentMethod_Bank transfer (automatic)'
]

# ========== Pydantic schema for raw customer data ==========
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# ========== Utility functions ==========
def preprocess_input(data):
    """
    Preprocess raw input data to match the exact preprocessing pipeline from training
    """
    df = pd.DataFrame([data])
    
    # 1. Label encode binary categorical columns (matching PDF preprocessing)
    binary_map = {'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0}
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    
    for col in binary_cols:
        df[col] = df[col].map(binary_map)
    
    # 2. One-hot encode multi-class categorical columns
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cols)
    
    # 3. Feature Engineering: Tenure binning (matching PDF conditions exactly)
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
    
    # One-hot encode tenure_range (matching PDF)
    df = pd.get_dummies(df, columns=['tenure_range'], prefix='tenure_group')
    
    # 4. Create log transformations (IMPORTANT - missing in original)
    df['TotalCharges_log'] = np.log1p(df['TotalCharges'])
    df['MonthlyCharges_log'] = np.log1p(df['MonthlyCharges'])
    
    # 5. Scale numerical features AFTER all feature engineering
    # The scaler was fitted on all three columns together: ['tenure', 'MonthlyCharges', 'TotalCharges']
    scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Apply the pre-fitted scaler to all three columns at once
    df[scale_cols] = scaler.transform(df[scale_cols])
    
    # 6. Select only the top features used in the final model
    # Ensure all required columns exist (fill missing with 0)
    for col in top_features:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the top features
    df_final = df[top_features]
    
    return df_final

# ========== API Endpoints ==========
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is live and ready."}

@app.post("/predict")
def predict_churn(customer: CustomerInput):
    """
    Predict customer churn based on raw input data
    """
    try:
        input_data = customer.dict()
        
        # Preprocess the input data
        processed_data = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        prediction_label = "Churn" if int(prediction) == 1 else "No Churn"

        return {
            "prediction": int(prediction),
            "churn_probability": float(round(probability, 4)),
            "prediction_label": prediction_label
        }

    
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "prediction": None,
            "churn_probability": None
        }

@app.get("/")
def root():
    return {"message": "Telco Customer Churn Prediction API", "docs": "/docs"}

# Example usage endpoint for testing
@app.get("/example")
def get_example_input():
    """
    Returns an example input for testing the API
    """
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 50.05,
        "TotalCharges": 1500.50
    }