# 📞 Telco Customer Churn Prediction


A comprehensive machine learning project that predicts customer churn for telecommunication companies using advanced ensemble methods and an interactive web application, developed as a part of Celebal Summer Internship 2025.

## 🚀 Live Demo

**Web Application:** [https://customerchurnpredictiontelco.streamlit.app/](https://customerchurnpredictiontelco.streamlit.app/)

## 👨‍💻 Project Information

**Name:** Disha Mondal  
**Email ID:** dishajhinuk4@gmail.com  
**Student ID:** CT_CSI_DS_5148  
**Domain:** Data Science  
**Organization:** Celebal Technologies

## 📊 Dataset

**Source:** [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)

The dataset contains information about a fictional telco company's customers and their churn status. It includes 7,043 customer records with 21 features covering demographics, account information, and services subscribed.

## 🎯 Project Overview

Customer churn prediction is crucial for telecommunications companies to:
- Identify at-risk customers before they leave
- Implement targeted retention strategies
- Reduce customer acquisition costs
- Improve overall customer lifetime value

This project implements a complete machine learning pipeline from data preprocessing to deployment, achieving **85.25% accuracy** with a Random Forest model and **84.74% accuracy** with an advanced Stacking Classifier.

## 🛠️ Technical Architecture

### Data Pipeline
1. **Data Loading & Exploration**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering & Preprocessing**
4. **Model Training & Evaluation**
5. **Web Application Development**
6. **Deployment**

### Key Features
- **Advanced Feature Engineering:** Tenure binning, log transformations, and scaling
- **Class Imbalance Handling:** SMOTE (Synthetic Minority Oversampling Technique)
- **Feature Selection:** Random Forest-based importance ranking
- **Ensemble Methods:** Stacking, Bagging, and Boosting
- **Interactive Web App:** Real-time predictions with beautiful visualizations

## 📈 Model Performance

### Best Performing Models

| Model | Accuracy | ROC AUC | F1 Score | Precision | Recall |
|-------|----------|---------|----------|-----------|--------|
| **Random Forest** | **85.25%** | **91.71%** | **85.52%** | **84.10%** | **86.98%** |
| Stacking Classifier | 84.74% | 91.41% | 85.01% | 83.60% | 86.46% |
| Advanced Stacking | 84.22% | 91.60% | 84.81% | 81.83% | 88.01% |
| AdaBoost | 79.77% | 87.63% | 80.58% | 77.53% | 83.88% |

### Model Components (Advanced Stacking)
- **Base Learners:** Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, MLP
- **Meta-Learner:** XGBoost
- **Cross-Validation:** 5-fold stratified

## 🔍 Key Insights from EDA

### Strong Churn Predictors
- **Contract Type:** Month-to-month customers churn significantly more (73% vs 11% for two-year)
- **Payment Method:** Electronic check users have highest churn rate
- **Services:** Customers without OnlineSecurity, TechSupport, and OnlineBackup churn more
- **Internet Service:** Fiber optic customers show higher churn rates
- **Tenure:** New customers (0-12 months) have highest churn risk

### Customer Behavior Patterns
- Customers with longer tenure are more loyal
- Higher monthly charges correlate with increased churn probability
- Family customers (with partners/dependents) are more stable
- Paperless billing users tend to churn more

## 🏗️ Implementation Details

### Data Preprocessing
```python
# Key preprocessing steps
1. Missing Value Imputation (TotalCharges: 11 missing values)
2. Label Encoding (Binary features: Gender, Partner, Dependents, etc.)
3. One-Hot Encoding (Multi-class features: Contract, PaymentMethod, etc.)
4. Feature Engineering (Tenure binning, Log transformations)
5. Standardization (StandardScaler for numerical features)
6. Class Balancing (SMOTE oversampling)
```

### Feature Engineering
- **Tenure Binning:** 6 groups (0-12, 12-24, 24-36, 36-48, 48-60, 60+ months)
- **Log Transformations:** Applied to TotalCharges and MonthlyCharges to reduce skewness
- **Feature Selection:** Top 18 most important features selected using Random Forest

### Model Training Strategy
1. **Train-Test Split:** 70% training, 30% testing
2. **Cross-Validation:** 5-fold for hyperparameter tuning
3. **Metrics:** Accuracy, ROC AUC, F1-Score, Precision, Recall
4. **Model Selection:** Based on balanced performance across all metrics

## 🖥️ Web Application Features

### Interactive Dashboard
- **Real-time Predictions:** Instant churn probability calculation
- **Risk Assessment:** Visual probability gauge (0-100%)
- **Risk Factor Analysis:** Identification of key churn drivers
- **Professional UI:** Clean, responsive design with custom CSS

### Input Categories
- **Demographics:** Gender, Senior Citizen status, Partner, Dependents
- **Account Information:** Tenure, Monthly/Total charges, Contract type
- **Services:** Phone, Internet, Security services, Streaming options
- **Billing:** Payment method, Paperless billing preference

### Visualization Components
- **Probability Gauge:** Color-coded risk indicator
- **Customer Profile:** Comprehensive information display
- **Risk Factors:** Actionable insights for retention strategies

## 📁 Project Screenshots
![HighRisk1](https://github.com/user-attachments/assets/68d7354d-be1c-4609-862b-7993e34f2155)
![HighRisk2](https://github.com/user-attachments/assets/e5c04ceb-e4cf-4567-8355-de665e1f46c0)
![LowRisk1](https://github.com/user-attachments/assets/2b2d9874-f480-4163-a173-a427c5347265)
![LowRisk2](https://github.com/user-attachments/assets/d94eb940-c85d-49bc-9943-6527537360e9)

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation
1. **Clone the repository**
```bash
git clone https://github.com/your-username/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run src/app.py
```

## 📊 Business Impact

### Retention Strategy Recommendations
1. **High-Risk Customers:** Proactive outreach for month-to-month contracts
2. **Service Bundling:** Promote security and support services to vulnerable segments
3. **Payment Incentives:** Encourage automatic payment methods
4. **Early Intervention:** Focus on customers in first 12 months
5. **Pricing Strategy:** Review pricing for high monthly charge customers

### Expected Outcomes
- **25-30% reduction** in customer churn rate
- **15-20% improvement** in customer lifetime value
- **Proactive retention** instead of reactive damage control
- **Data-driven decision making** for marketing campaigns


## 📄 Model Interpretability

### Business Rules
- Tenure < 12 months: High risk (+40% churn probability)
- Month-to-month + Electronic check: Very high risk (+60% churn probability)
- No security services + Fiber optic: Elevated risk (+25% churn probability)

## 🙏 Acknowledgments

- **Celebal Technologies** for the internship opportunity
- **Kaggle** for providing the dataset
- **Streamlit** for the amazing web framework
- **scikit-learn** and **XGBoost** teams for excellent ML libraries

---
