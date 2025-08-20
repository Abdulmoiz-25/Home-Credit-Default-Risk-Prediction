import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Loan Default Risk Prediction", layout="wide")

# Load models
@st.cache_resource
def load_models():
    try:
        log_model = joblib.load("log_model.pkl")
        scaler = joblib.load("scaler.pkl")
        cat_model = CatBoostClassifier()
        cat_model.load_model("cat_model.cbm")
        return log_model, scaler, cat_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

log_model, scaler, cat_model = load_models()

@st.cache_data
def create_sample_data():
    # Complete sample data with ALL Home Credit features (not just subset)
    sample_data = {
        'SK_ID_CURR': [100002, 100003, 100004, 100006, 100007],
        'NAME_CONTRACT_TYPE': ['Cash loans', 'Cash loans', 'Revolving loans', 'Cash loans', 'Cash loans'],
        'CODE_GENDER': ['M', 'F', 'M', 'F', 'M'],
        'FLAG_OWN_CAR': ['N', 'N', 'Y', 'N', 'N'],
        'FLAG_OWN_REALTY': ['Y', 'N', 'Y', 'Y', 'Y'],
        'CNT_CHILDREN': [0, 0, 0, 0, 0],
        'AMT_INCOME_TOTAL': [202500.0, 270000.0, 67500.0, 135000.0, 121500.0],
        'AMT_CREDIT': [406597.5, 1293502.5, 135000.0, 312682.5, 513000.0],
        'AMT_ANNUITY': [24700.5, 35698.5, 6750.0, 29686.5, 21865.5],
        'AMT_GOODS_PRICE': [351000.0, 1129500.0, 135000.0, 297000.0, 513000.0],
        'NAME_TYPE_SUITE': ['Unaccompanied', 'Family', 'Unaccompanied', 'Unaccompanied', 'Unaccompanied'],
        'NAME_INCOME_TYPE': ['Working', 'State servant', 'Working', 'Working', 'Working'],
        'NAME_EDUCATION_TYPE': ['Secondary / secondary special', 'Higher education', 'Secondary / secondary special', 'Secondary / secondary special', 'Secondary / secondary special'],
        'NAME_FAMILY_STATUS': ['Single / not married', 'Married', 'Single / not married', 'Civil marriage', 'Single / not married'],
        'NAME_HOUSING_TYPE': ['House / apartment', 'House / apartment', 'House / apartment', 'House / apartment', 'House / apartment'],
        'REGION_POPULATION_RELATIVE': [0.018801, 0.003541, 0.010032, 0.008019, 0.028663],
        'DAYS_BIRTH': [-9461, -16765, -19046, -19005, -19932],
        'DAYS_EMPLOYED': [-637, -1188, -225, -3039, -3038],
        'DAYS_REGISTRATION': [-3648.0, -1186.0, -4260.0, -9833.0, -4311.0],
        'DAYS_ID_PUBLISH': [-2120, -291, -2531, -2437, -3458],
        'OWN_CAR_AGE': [np.nan, np.nan, 26.0, np.nan, np.nan],
        'FLAG_MOBIL': [1, 1, 1, 1, 1],
        'FLAG_EMP_PHONE': [1, 1, 1, 1, 1],
        'FLAG_WORK_PHONE': [0, 0, 1, 0, 0],
        'FLAG_CONT_MOBILE': [1, 1, 1, 1, 1],
        'FLAG_PHONE': [1, 1, 1, 0, 0],
        'FLAG_EMAIL': [0, 0, 0, 0, 0],
        'OCCUPATION_TYPE': ['Laborers', 'Core staff', 'Laborers', 'Laborers', 'Core staff'],
        'CNT_FAM_MEMBERS': [1.0, 2.0, 1.0, 2.0, 1.0],
        'REGION_RATING_CLIENT': [2, 1, 2, 2, 2],
        'REGION_RATING_CLIENT_W_CITY': [2, 1, 2, 2, 2],
        'WEEKDAY_APPR_PROCESS_START': ['WEDNESDAY', 'MONDAY', 'MONDAY', 'WEDNESDAY', 'THURSDAY'],
        'HOUR_APPR_PROCESS_START': [10, 11, 9, 17, 11],
        'REG_REGION_NOT_LIVE_REGION': [0, 0, 0, 0, 0],
        'REG_REGION_NOT_WORK_REGION': [0, 0, 0, 0, 0],
        'LIVE_REGION_NOT_WORK_REGION': [0, 0, 0, 0, 0],
        'REG_CITY_NOT_LIVE_CITY': [0, 0, 0, 0, 0],
        'REG_CITY_NOT_WORK_CITY': [0, 0, 0, 0, 0],
        'LIVE_CITY_NOT_WORK_CITY': [0, 0, 0, 0, 0],
        'ORGANIZATION_TYPE': ['Business Entity Type 3', 'School', 'Government', 'Business Entity Type 3', 'Religion'],
        'EXT_SOURCE_1': [0.083037, 0.311267, np.nan, np.nan, np.nan],
        'EXT_SOURCE_2': [0.262949, 0.622246, 0.555912, 0.650442, 0.322738],
        'EXT_SOURCE_3': [0.139376, np.nan, 0.729567, np.nan, np.nan],
        'APARTMENTS_AVG': [0.0149, np.nan, 0.0714, np.nan, np.nan],
        'BASEMENTAREA_AVG': [0.0714, np.nan, np.nan, np.nan, np.nan],
        'YEARS_BEGINEXPLUATATION_AVG': [0.7652, np.nan, 0.5085, np.nan, np.nan],
        'YEARS_BUILD_AVG': [0.7738, np.nan, 0.3571, np.nan, np.nan],
        'COMMONAREA_AVG': [0.0714, np.nan, np.nan, np.nan, np.nan],
        'ELEVATORS_AVG': [0.0000, np.nan, 0.0714, np.nan, np.nan],
        'ENTRANCES_AVG': [0.0714, np.nan, 0.0714, np.nan, np.nan],
        'FLOORSMAX_AVG': [0.1429, np.nan, 0.0714, np.nan, np.nan],
        'FLOORSMIN_AVG': [0.1429, np.nan, 0.0714, np.nan, np.nan],
        'LANDAREA_AVG': [0.0714, np.nan, np.nan, np.nan, np.nan],
        'LIVINGAPARTMENTS_AVG': [0.0149, np.nan, 0.0714, np.nan, np.nan],
        'LIVINGAREA_AVG': [0.0714, np.nan, 0.0714, np.nan, np.nan],
        'NONLIVINGAPARTMENTS_AVG': [0.0000, np.nan, 0.0000, np.nan, np.nan],
        'NONLIVINGAREA_AVG': [0.0000, np.nan, 0.0000, np.nan, np.nan],
        'AMT_REQ_CREDIT_BUREAU_HOUR': [0.0, 0.0, 0.0, 0.0, 0.0],
        'AMT_REQ_CREDIT_BUREAU_DAY': [0.0, 0.0, 0.0, 0.0, 0.0],
        'AMT_REQ_CREDIT_BUREAU_WEEK': [0.0, 0.0, 0.0, 0.0, 0.0],
        'AMT_REQ_CREDIT_BUREAU_MON': [0.0, 0.0, 0.0, 0.0, 0.0],
        'AMT_REQ_CREDIT_BUREAU_QRT': [0.0, 0.0, 0.0, 0.0, 0.0],
        'AMT_REQ_CREDIT_BUREAU_YEAR': [1.0, 1.0, 0.0, 0.0, 0.0],
        'APARTMENTS_MODE': [0.0149, np.nan, 0.0714, np.nan, np.nan],
        'BASEMENTAREA_MODE': [0.0714, np.nan, np.nan, np.nan, np.nan],
        'YEARS_BEGINEXPLUATATION_MODE': [0.7652, np.nan, 0.5085, np.nan, np.nan],
        'YEARS_BUILD_MODE': [0.7738, np.nan, 0.3571, np.nan, np.nan],
        'COMMONAREA_MODE': [0.0714, np.nan, np.nan, np.nan, np.nan],
        'ELEVATORS_MODE': [0.0000, np.nan, 0.0714, np.nan, np.nan],
        'ENTRANCES_MODE': [0.0714, np.nan, 0.0714, np.nan, np.nan],
        'FLOORSMAX_MODE': [0.1429, np.nan, 0.0714, np.nan, np.nan],
        'FLOORSMIN_MODE': [0.1429, np.nan, 0.0714, np.nan, np.nan],
        'LANDAREA_MODE': [0.0714, np.nan, np.nan, np.nan, np.nan],
        'LIVINGAPARTMENTS_MODE': [0.0149, np.nan, 0.0714, np.nan, np.nan],
        'LIVINGAREA_MODE': [0.0714, np.nan, 0.0714, np.nan, np.nan],
        'NONLIVINGAPARTMENTS_MODE': [0.0000, np.nan, 0.0000, np.nan, np.nan],
        'NONLIVINGAREA_MODE': [0.0000, np.nan, 0.0000, np.nan, np.nan],
        'APARTMENTS_MEDI': [0.0149, np.nan, 0.0714, np.nan, np.nan],
        'BASEMENTAREA_MEDI': [0.0714, np.nan, np.nan, np.nan, np.nan],
        'YEARS_BEGINEXPLUATATION_MEDI': [0.7652, np.nan, 0.5085, np.nan, np.nan],
        'YEARS_BUILD_MEDI': [0.7738, np.nan, 0.3571, np.nan, np.nan],
        'COMMONAREA_MEDI': [0.0714, np.nan, np.nan, np.nan, np.nan],
        'ELEVATORS_MEDI': [0.0000, np.nan, 0.0714, np.nan, np.nan],
        'ENTRANCES_MEDI': [0.0714, np.nan, 0.0714, np.nan, np.nan],
        'FLOORSMAX_MEDI': [0.1429, np.nan, 0.0714, np.nan, np.nan],
        'FLOORSMIN_MEDI': [0.1429, np.nan, 0.0714, np.nan, np.nan],
        'LANDAREA_MEDI': [0.0714, np.nan, np.nan, np.nan, np.nan],
        'LIVINGAPARTMENTS_MEDI': [0.0149, np.nan, 0.0714, np.nan, np.nan],
        'LIVINGAREA_MEDI': [0.0714, np.nan, 0.0714, np.nan, np.nan],
        'NONLIVINGAPARTMENTS_MEDI': [0.0000, np.nan, 0.0000, np.nan, np.nan],
        'NONLIVINGAREA_MEDI': [0.0000, np.nan, 0.0000, np.nan, np.nan],
        'FLAG_DOCUMENT_2': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_3': [1, 1, 1, 1, 1],
        'FLAG_DOCUMENT_4': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_5': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_6': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_7': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_8': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_9': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_10': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_11': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_12': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_13': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_14': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_15': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_16': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_17': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_18': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_19': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_20': [0, 0, 0, 0, 0],
        'FLAG_DOCUMENT_21': [0, 0, 0, 0, 0],
        'TARGET': [1, 0, 0, 0, 0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Apply exact same preprocessing as Colab training
    # 1. Handle missing values (numeric columns) - fill with median
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # 2. Encode categorical variables (exactly like training)
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # 3. Split features and target
    X = df_encoded.drop('TARGET', axis=1)
    
    return X

# Get training feature structure
training_features = create_sample_data()

st.title("🏦 Loan Default Risk Prediction")
st.markdown("Predict loan default risk using Logistic Regression and CatBoost models trained on Home Credit data.")

# Sidebar
st.sidebar.header("Prediction Settings")
model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "CatBoost", "Both"])

# Main interface
st.header("Enter Applicant Details")

if log_model is None or scaler is None or cat_model is None:
    st.error("Models could not be loaded. Please check model files.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Information")
    amt_income_total = st.number_input("Annual Income", value=202500.0, min_value=0.0)
    amt_credit = st.number_input("Credit Amount", value=406597.5, min_value=0.0)
    amt_annuity = st.number_input("Loan Annuity", value=24700.5, min_value=0.0)
    amt_goods_price = st.number_input("Goods Price", value=351000.0, min_value=0.0)
    
    st.subheader("Personal Information")
    cnt_children = st.number_input("Number of Children", value=0, min_value=0, max_value=20)
    days_birth = st.number_input("Age (years)", value=26, min_value=18, max_value=100)
    days_employed = st.number_input("Years Employed", value=2, min_value=0, max_value=50)

with col2:
    st.subheader("Categorical Information")
    code_gender = st.selectbox("Gender", ["M", "F"])
    name_contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
    flag_own_car = st.selectbox("Owns Car", ["N", "Y"])
    flag_own_realty = st.selectbox("Owns Realty", ["Y", "N"])
    name_income_type = st.selectbox("Income Type", ["Working", "State servant", "Commercial associate", "Pensioner"])
    name_education_type = st.selectbox("Education Level", 
                                     ["Secondary / secondary special", "Higher education", 
                                      "Incomplete higher", "Lower secondary", "Academic degree"])
    name_family_status = st.selectbox("Family Status", 
                                    ["Single / not married", "Married", "Civil marriage", 
                                     "Separated", "Widow"])

if st.button("🔮 Predict Default Risk", type="primary"):
    # Create new applicant data with same structure as training
    new_applicant = {
        'SK_ID_CURR': 999999,  # Dummy ID
        'NAME_CONTRACT_TYPE': name_contract_type,
        'CODE_GENDER': code_gender,
        'FLAG_OWN_CAR': flag_own_car,
        'FLAG_OWN_REALTY': flag_own_realty,
        'CNT_CHILDREN': cnt_children,
        'AMT_INCOME_TOTAL': amt_income_total,
        'AMT_CREDIT': amt_credit,
        'AMT_ANNUITY': amt_annuity,
        'AMT_GOODS_PRICE': amt_goods_price,
        'NAME_TYPE_SUITE': 'Unaccompanied',
        'NAME_INCOME_TYPE': name_income_type,
        'NAME_EDUCATION_TYPE': name_education_type,
        'NAME_FAMILY_STATUS': name_family_status,
        'NAME_HOUSING_TYPE': 'House / apartment',
        'REGION_POPULATION_RELATIVE': 0.018801,
        'DAYS_BIRTH': -days_birth * 365,  # Convert to negative days
        'DAYS_EMPLOYED': -days_employed * 365,  # Convert to negative days
        'DAYS_REGISTRATION': -3648.0,
        'DAYS_ID_PUBLISH': -2120,
        'OWN_CAR_AGE': np.nan,
        'FLAG_MOBIL': 1,
        'FLAG_EMP_PHONE': 1,
        'FLAG_WORK_PHONE': 0,
        'FLAG_CONT_MOBILE': 1,
        'FLAG_PHONE': 1,
        'FLAG_EMAIL': 0,
        'OCCUPATION_TYPE': 'Laborers',
        'CNT_FAM_MEMBERS': float(cnt_children + 1),
        'REGION_RATING_CLIENT': 2,
        'REGION_RATING_CLIENT_W_CITY': 2,
        'WEEKDAY_APPR_PROCESS_START': 'WEDNESDAY',
        'HOUR_APPR_PROCESS_START': 10,
        'REG_REGION_NOT_LIVE_REGION': 0,
        'REG_REGION_NOT_WORK_REGION': 0,
        'LIVE_REGION_NOT_WORK_REGION': 0,
        'REG_CITY_NOT_LIVE_CITY': 0,
        'REG_CITY_NOT_WORK_CITY': 0,
        'LIVE_CITY_NOT_WORK_CITY': 0,
        'ORGANIZATION_TYPE': 'Business Entity Type 3',
        'EXT_SOURCE_1': 0.083037,
        'EXT_SOURCE_2': 0.262949,
        'EXT_SOURCE_3': 0.139376,
        'APARTMENTS_AVG': 0.0149,
        'BASEMENTAREA_AVG': 0.0714,
        'YEARS_BEGINEXPLUATATION_AVG': 0.7652,
        'YEARS_BUILD_AVG': 0.7738,
        'COMMONAREA_AVG': 0.0714,
        'ELEVATORS_AVG': 0.0000,
        'ENTRANCES_AVG': 0.0714,
        'FLOORSMAX_AVG': 0.1429,
        'FLOORSMIN_AVG': 0.1429,
        'LANDAREA_AVG': 0.0714,
        'LIVINGAPARTMENTS_AVG': 0.0149,
        'LIVINGAREA_AVG': 0.0714,
        'NONLIVINGAPARTMENTS_AVG': 0.0000,
        'NONLIVINGAREA_AVG': 0.0000,
        'AMT_REQ_CREDIT_BUREAU_HOUR': 0.0,
        'AMT_REQ_CREDIT_BUREAU_DAY': 0.0,
        'AMT_REQ_CREDIT_BUREAU_WEEK': 0.0,
        'AMT_REQ_CREDIT_BUREAU_MON': 0.0,
        'AMT_REQ_CREDIT_BUREAU_QRT': 0.0,
        'AMT_REQ_CREDIT_BUREAU_YEAR': 1.0,
        'APARTMENTS_MODE': 0.0149,
        'BASEMENTAREA_MODE': 0.0714,
        'YEARS_BEGINEXPLUATATION_MODE': 0.7652,
        'YEARS_BUILD_MODE': 0.7738,
        'COMMONAREA_MODE': 0.0714,
        'ELEVATORS_MODE': 0.0000,
        'ENTRANCES_MODE': 0.0714,
        'FLOORSMAX_MODE': 0.1429,
        'FLOORSMIN_MODE': 0.1429,
        'LANDAREA_MODE': 0.0714,
        'LIVINGAPARTMENTS_MODE': 0.0149,
        'LIVINGAREA_MODE': 0.0714,
        'NONLIVINGAPARTMENTS_MODE': 0.0000,
        'NONLIVINGAREA_MODE': 0.0000,
        'APARTMENTS_MEDI': 0.0149,
        'BASEMENTAREA_MEDI': 0.0714,
        'YEARS_BEGINEXPLUATATION_MEDI': 0.7652,
        'YEARS_BUILD_MEDI': 0.7738,
        'COMMONAREA_MEDI': 0.0714,
        'ELEVATORS_MEDI': 0.0000,
        'ENTRANCES_MEDI': 0.0714,
        'FLOORSMAX_MEDI': 0.1429,
        'FLOORSMIN_MEDI': 0.1429,
        'LANDAREA_MEDI': 0.0714,
        'LIVINGAPARTMENTS_MEDI': 0.0149,
        'LIVINGAREA_MEDI': 0.0714,
        'NONLIVINGAPARTMENTS_MEDI': 0.0000,
        'NONLIVINGAREA_MEDI': 0.0000,
        'FLAG_DOCUMENT_2': 0,
        'FLAG_DOCUMENT_3': 1,
        'FLAG_DOCUMENT_4': 0,
        'FLAG_DOCUMENT_5': 0,
        'FLAG_DOCUMENT_6': 0,
        'FLAG_DOCUMENT_7': 0,
        'FLAG_DOCUMENT_8': 0,
        'FLAG_DOCUMENT_9': 0,
        'FLAG_DOCUMENT_10': 0,
        'FLAG_DOCUMENT_11': 0,
        'FLAG_DOCUMENT_12': 0,
        'FLAG_DOCUMENT_13': 0,
        'FLAG_DOCUMENT_14': 0,
        'FLAG_DOCUMENT_15': 0,
        'FLAG_DOCUMENT_16': 0,
        'FLAG_DOCUMENT_17': 0,
        'FLAG_DOCUMENT_18': 0,
        'FLAG_DOCUMENT_19': 0,
        'FLAG_DOCUMENT_20': 0,
        'FLAG_DOCUMENT_21': 0,
        'TARGET': 0  # Dummy target
    }
    
    # Convert to DataFrame
    new_df = pd.DataFrame([new_applicant])
    
    # Apply exact same preprocessing as training
    # 1. Handle missing values (numeric columns)
    numeric_cols = new_df.select_dtypes(include=['number']).columns
    new_df[numeric_cols] = new_df[numeric_cols].fillna(new_df[numeric_cols].median())
    
    # 2. Apply one-hot encoding (exactly like training)
    new_df_encoded = pd.get_dummies(new_df, drop_first=True)
    
    # 3. Remove target column
    X_new = new_df_encoded.drop('TARGET', axis=1)
    
    # 4. Align with training features (reindex to match exactly)
    X_new_aligned = X_new.reindex(columns=training_features.columns, fill_value=0)
    
    st.write(f"Debug: Input shape: {X_new_aligned.shape}")
    st.write(f"Debug: Expected shape: {training_features.shape}")
    
    # Make predictions
    results = {}
    
    if model_choice in ["Logistic Regression", "Both"]:
        try:
            # For Logistic Regression: use scaled features
            X_scaled = scaler.transform(X_new_aligned)
            lr_prob = log_model.predict_proba(X_scaled)[0, 1]
            results["Logistic Regression"] = lr_prob
        except Exception as e:
            st.error(f"Logistic Regression error: {e}")
    
    if model_choice in ["CatBoost", "Both"]:
        try:
            # For CatBoost: use original features (no scaling)
            cb_prob = cat_model.predict_proba(X_new_aligned)[0, 1]
            results["CatBoost"] = cb_prob
        except Exception as e:
            st.error(f"CatBoost error: {e}")
    
    # Display results
    if results:
        st.success("✅ Prediction completed!")
        
        for model_name, probability in results.items():
            risk_level = "🔴 HIGH RISK" if probability > 0.5 else "🟢 LOW RISK"
            st.metric(
                label=f"{model_name} Prediction",
                value=f"{probability:.1%}",
                delta=risk_level
            )
            
            # Progress bar for visual representation
            st.progress(probability)
        
        # Business interpretation
        st.subheader("Business Interpretation")
        avg_prob = np.mean(list(results.values()))
        
        if avg_prob > 0.7:
            st.error("⚠️ **REJECT LOAN** - Very high default risk")
        elif avg_prob > 0.5:
            st.warning("⚠️ **REVIEW CAREFULLY** - Moderate to high default risk")
        elif avg_prob > 0.3:
            st.info("ℹ️ **APPROVE WITH CONDITIONS** - Low to moderate default risk")
        else:
            st.success("✅ **APPROVE LOAN** - Low default risk")
    else:
        st.error("❌ No predictions could be generated. Please check the model files.")

st.markdown("---")
st.markdown("**Note:** This app uses models trained on Home Credit dataset with business cost optimization.")
