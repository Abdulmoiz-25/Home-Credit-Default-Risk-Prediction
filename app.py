import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import zipfile
import os
import json

st.set_page_config(page_title="Loan Default Risk Prediction", layout="wide")

@st.cache_data
def load_training_dataset():
    """Load the actual training dataset from uploaded zip file"""
    try:
        # Check if zip file exists in the repo
        zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]

        with st.sidebar.expander("📊 Dataset Status", expanded=True):
            if zip_files:
                zip_file = zip_files[0]  # Use the first zip file found
                st.markdown(f"📦 Found dataset: **{zip_file}**")

                # Extract the zip file
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall('.')

                # Look for CSV files
                csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'train' in f.lower()]

                if csv_files:
                    csv_file = csv_files[0]  # Use the first training CSV found
                    st.markdown(f"📂 Using training data: **{csv_file}**")

                    # Load the actual training dataset
                    df = pd.read_csv(csv_file)

                    # Apply exact same preprocessing as Colab training
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

                    df_encoded = pd.get_dummies(df, drop_first=True)

                    if 'TARGET' in df_encoded.columns:
                        X = df_encoded.drop('TARGET', axis=1)
                    else:
                        X = df_encoded

                    st.markdown(f"✅ Training dataset loaded ({X.shape[0]} rows, {X.shape[1]} features)")
                    return X
                else:
                    st.markdown("⚠️ No training CSV file found in zip — using fallback sample data.")
                    return create_sample_data()
            else:
                st.markdown("⚠️ No zip file found — using fallback sample data.")
                return create_sample_data()

    except Exception as e:
        with st.sidebar.expander("📊 Dataset Status", expanded=True):
            st.markdown(f"❌ Error loading training dataset: `{e}`")
            st.markdown("⚠️ Using fallback sample data.")
        return create_sample_data()


# Load models
st.set_page_config(page_title="Loan Default Risk Prediction", layout="wide")

@st.cache_data
def load_training_dataset():
    """Load the actual training dataset from uploaded zip file"""
    # Always open only ONE expander
    with st.sidebar.expander("📊 Dataset Status", expanded=False):  
        try:
            # Check if zip file exists in the repo
            zip_files = [f for f in os.listdir('.') if f.endswith('.zip')]

            if zip_files:
                zip_file = zip_files[0]  # Use the first zip file found
                st.markdown(f"📦 Found dataset: **{zip_file}**")

                # Extract the zip file
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall('.')

                # Look for CSV files
                csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'train' in f.lower()]

                if csv_files:
                    csv_file = csv_files[0]  # Use the first training CSV found
                    st.markdown(f"📂 Using training data: **{csv_file}**")

                    # Load the actual training dataset
                    df = pd.read_csv(csv_file)

                    # Apply exact same preprocessing as Colab training
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

                    df_encoded = pd.get_dummies(df, drop_first=True)

                    if 'TARGET' in df_encoded.columns:
                        X = df_encoded.drop('TARGET', axis=1)
                    else:
                        X = df_encoded

                    st.markdown(f"✅ Training dataset loaded ({X.shape[0]} rows, {X.shape[1]} features)")
                    return X
                else:
                    st.markdown("⚠️ No training CSV file found in zip — using fallback sample data.")
                    return create_sample_data()
            else:
                st.markdown("⚠️ No zip file found — using fallback sample data.")
                return create_sample_data()

        except Exception as e:
            st.markdown(f"❌ Error loading training dataset: `{e}`")
            st.markdown("⚠️ Using fallback sample data.")
            return create_sample_data()

@st.cache_resource
def load_models():
    with st.sidebar.expander("🤖 Model Status", expanded=False):  # collapsed by default
        try:
            log_model = joblib.load("log_model.pkl")
            st.markdown("✅ Logistic Regression model loaded")

            scaler = joblib.load("scaler.pkl")
            st.markdown("✅ Scaler loaded")

            cat_model = CatBoostClassifier()
            cat_model.load_model("cat_model.cbm")
            st.markdown("✅ CatBoost model loaded")

            return log_model, scaler, cat_model
        except Exception as e:
            st.markdown(f"❌ Error loading models: `{e}`")
            return None, None, None

log_model, scaler, cat_model = load_models()



@st.cache_data
def create_sample_data():
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

training_features = load_training_dataset()

# --- App Header ---
st.title("🏦 Loan Default Risk Prediction")
st.markdown(
    "Easily predict the likelihood of loan default using "
    "**Logistic Regression** and **CatBoost** models trained on Home Credit data."
)

st.info(f"📊 Using training dataset with **{training_features.shape[1]} features** for feature alignment.")

# --- Sidebar Settings ---
with st.sidebar:
    st.header("⚙️ Prediction Settings")
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "CatBoost", "Both"])

# --- Applicant Section ---
st.header("📝 Applicant Information")
st.markdown("Fill in the applicant’s details below:")

if log_model is None or scaler is None or cat_model is None:
    st.error("❌ Models could not be loaded. Please check model files.")
    st.stop()

# --- Input Layout ---
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💰 Financial Information")
        amt_income_total = st.number_input("Annual Income ($)", value=202500.0, min_value=0.0)
        amt_credit = st.number_input("Credit Amount ($)", value=406597.5, min_value=0.0)
        amt_annuity = st.number_input("Loan Annuity ($)", value=24700.5, min_value=0.0)
        amt_goods_price = st.number_input("Goods Price ($)", value=351000.0, min_value=0.0)

        st.markdown("### 👤 Personal Information")
        cnt_children = st.number_input("Number of Children", value=0, min_value=0, max_value=20)
        days_birth = st.number_input("Age (years)", value=26, min_value=18, max_value=100)
        days_employed = st.number_input("Years Employed", value=2, min_value=0, max_value=50)

    with col2:
        st.markdown("### 📊 Categorical Information")
        code_gender = st.selectbox("Gender", ["M", "F"])
        name_contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
        flag_own_car = st.selectbox("Owns Car", ["N", "Y"])
        flag_own_realty = st.selectbox("Owns Realty", ["Y", "N"])
        name_income_type = st.selectbox(
            "Income Type",
            ["Working", "State servant", "Commercial associate", "Pensioner"]
        )
        name_education_type = st.selectbox(
            "Education Level",
            ["Secondary / secondary special", "Higher education",
             "Incomplete higher", "Lower secondary", "Academic degree"]
        )
        name_family_status = st.selectbox(
            "Family Status",
            ["Single / not married", "Married", "Civil marriage", "Separated", "Widow"]
        )

# Initialize results globally
results = {}

# --- Predict Button ---
if st.button("🔮 Predict Default Risk", type="primary"):

    # -------------------------
    # 1️⃣ Prepare new applicant data
    # -------------------------
    new_applicant = {
        'SK_ID_CURR': 999999,
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
        'DAYS_BIRTH': -days_birth * 365,
        'DAYS_EMPLOYED': -days_employed * 365,
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
        'ELEVATORS_AVG': 0.0,
        'ENTRANCES_AVG': 0.0714,
        'FLOORSMAX_AVG': 0.1429,
        'FLOORSMIN_AVG': 0.1429,
        'LANDAREA_AVG': 0.0714,
        'LIVINGAPARTMENTS_AVG': 0.0149,
        'LIVINGAREA_AVG': 0.0714,
        'NONLIVINGAPARTMENTS_AVG': 0.0,
        'NONLIVINGAREA_AVG': 0.0,
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
        'ELEVATORS_MODE': 0.0,
        'ENTRANCES_MODE': 0.0714,
        'FLOORSMAX_MODE': 0.1429,
        'FLOORSMIN_MODE': 0.1429,
        'LANDAREA_MODE': 0.0714,
        'LIVINGAPARTMENTS_MODE': 0.0149,
        'LIVINGAREA_MODE': 0.0714,
        'NONLIVINGAPARTMENTS_MODE': 0.0,
        'NONLIVINGAREA_MODE': 0.0,
        'APARTMENTS_MEDI': 0.0149,
        'BASEMENTAREA_MEDI': 0.0714,
        'YEARS_BEGINEXPLUATATION_MEDI': 0.7652,
        'YEARS_BUILD_MEDI': 0.7738,
        'COMMONAREA_MEDI': 0.0714,
        'ELEVATORS_MEDI': 0.0,
        'ENTRANCES_MEDI': 0.0714,
        'FLOORSMAX_MEDI': 0.1429,
        'FLOORSMIN_MEDI': 0.1429,
        'LANDAREA_MEDI': 0.0714,
        'LIVINGAPARTMENTS_MEDI': 0.0149,
        'LIVINGAREA_MEDI': 0.0714,
        'NONLIVINGAPARTMENTS_MEDI': 0.0,
        'NONLIVINGAREA_MEDI': 0.0,
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
        'TARGET': 0
    }

    new_df = pd.DataFrame([new_applicant])

    # -------------------------
    # 2️⃣ Preprocess input
    # -------------------------
    try:
        training_medians = training_features.median()
    except:
        training_medians = new_df.select_dtypes(include=['number']).median()

    for col in new_df.select_dtypes(include=['number']).columns:
        new_df[col] = new_df[col].fillna(training_medians.get(col, 0))

    new_df_encoded = pd.get_dummies(new_df, drop_first=True)
    X_new = new_df_encoded.drop('TARGET', axis=1, errors='ignore')
    X_new_aligned = X_new.reindex(columns=training_features.columns, fill_value=0)
    X_new_aligned = X_new_aligned.fillna(0)

    # -------------------------
    # 3️⃣ Make predictions
    # -------------------------
    results = {}

    if model_choice in ["Logistic Regression", "Both"]:
        try:
            X_scaled = scaler.transform(X_new_aligned)
            lr_prob = log_model.predict_proba(X_scaled)[0, 1]
            results["Logistic Regression"] = lr_prob
        except Exception as e:
            st.error(f"Logistic Regression error: {e}")

    if model_choice in ["CatBoost", "Both"]:
        try:
            cb_prob = cat_model.predict_proba(X_new_aligned)[0, 1]
            results["CatBoost"] = cb_prob
        except Exception as e:
            st.error(f"CatBoost error: {e}")

# -------------------------
# 4️⃣ Display results (cards + animated circular gauge)
# -------------------------
def prob_color(prob):
    if prob > 0.7: return "#d9534f"
    elif prob > 0.5: return "#f39c12"
    elif prob > 0.3: return "#3498db"
    else: return "#2ecc71"

def risk_badge(prob):
    if prob > 0.7: return "High Risk"
    elif prob > 0.5: return "Moderate Risk"
    elif prob > 0.3: return "Low Risk"
    else: return "Very Low Risk"

def circular_gauge_svg(pct, size=160, color="#00e676", risk_text="Low Risk"):
    radius = size / 2 - 12
    circumference = 2 * 3.1415 * radius
    offset = circumference * (1 - pct / 100)
    svg = f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <defs>
        <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style="stop-color:{color};stop-opacity:1" />
          <stop offset="100%" style="stop-color:#ffffff;stop-opacity:0.1" />
        </linearGradient>
      </defs>
      <circle cx="{size/2}" cy="{size/2}" r="{radius}" stroke="#2d3748" stroke-width="12" fill="none"/>
      <circle cx="{size/2}" cy="{size/2}" r="{radius}" stroke="url(#grad)" stroke-width="12" fill="none"
              stroke-dasharray="{circumference}" stroke-dashoffset="{circumference}"
              transform="rotate(-90 {size/2} {size/2})">
        <animate attributeName="stroke-dashoffset" from="{circumference}" to="{offset}" dur="1.2s" fill="freeze" />
      </circle>
      <text x="50%" y="50%" text-anchor="middle" dy="7" fill="white" font-size="20">{pct:.1f}%</text>
      <title>{risk_text}</title>
    </svg>
    """
    return svg

if results:
    # --- Display model cards ---
    st.markdown("<h3 style='color:white;'>✅ Prediction completed!</h3>", unsafe_allow_html=True)
    n = len(results)
    cols = st.columns(n)
    for (model_name, probability), col in zip(results.items(), cols):
        pct = float(probability)
        color = prob_color(pct)
        badge = risk_badge(pct)
        pct_display = f"{pct:.1%}"
        bar_width = int(pct * 100)
        gradient_css = f"background: linear-gradient(90deg, {color}, rgba(255,255,255,0.06));"
        card_html = f"""
        <div style='border-radius:12px;padding:18px;background:#1f2937;margin-bottom:18px;box-shadow:0 6px 18px rgba(8,12,20,0.6);'>
          <div style='font-size:16px;color:#cbd5e1;margin-bottom:6px;'>{model_name} Prediction</div>
          <div style='font-size:44px;font-weight:800;margin:4px 0 8px 0;color:{color};'>{pct_display}</div>
          <div style="margin-bottom:8px;">
            <span style="background:{color};color:#fff;padding:6px 10px;border-radius:999px;font-size:13px;">{badge}</span>
            <span style="margin-left:10px;font-size:13px;color:#9aa7b8;">Probability of default</span>
          </div>
          <div style="width:100%;height:10px;background: rgba(255,255,255,0.06);border-radius:999px;margin-top:12px;">
            <div style="height:100%;border-radius:999px;{gradient_css} width:{bar_width}%;"></div>
          </div>
        </div>
        """
        col.markdown(card_html, unsafe_allow_html=True)

    # --- Circular gauge + summary ---
    avg_prob = np.mean(list(results.values()))
    avg_pct = float(avg_prob * 100)
    if avg_prob > 0.7:
        recommendation = "⚠️ REJECT LOAN — Very high default risk"
        rec_color = "#d9534f"
        risk_text = "High Risk"
    elif avg_prob > 0.5:
        recommendation = "⚠️ REVIEW CAREFULLY — Moderate to high default risk"
        rec_color = "#f39c12"
        risk_text = "Moderate Risk"
    elif avg_prob > 0.3:
        recommendation = "ℹ️ APPROVE WITH CONDITIONS — Low to moderate default risk"
        rec_color = "#3498db"
        risk_text = "Low Risk"
    else:
        recommendation = "✅ APPROVE LOAN — Low default risk"
        rec_color = "#2ecc71"
        risk_text = "Very Low Risk"

    c1, c2 = st.columns([1,2])
    with c1:
        st.markdown(circular_gauge_svg(avg_pct, size=160, color=rec_color, risk_text=risk_text), unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center;color:#b9c6d4;'>Average probability</div>", unsafe_allow_html=True)

    with c2:
        rec_html = f"""
        <div style='border-radius:10px;padding:14px;margin-top:18px;background:#111827;box-shadow:0 8px 20px rgba(0,0,0,0.45);'>
          <div style='font-size:20px;font-weight:800;color:{rec_color};margin-bottom:6px;'>{recommendation}</div>
          <div style='color:#94a3b8;font-size:13px;'>Average probability across selected models: <strong style='color:white;'>{avg_prob:.1%}</strong></div>
        </div>
        """
        st.markdown(rec_html, unsafe_allow_html=True)

        json_blob = json.dumps({k: float(v) for k,v in results.items()}, indent=2)
        st.download_button("📥 Download predictions (JSON)", data=json_blob, file_name="prediction_results.json", mime="application/json")

else:
    st.error("❌ No predictions could be generated. Please check the model files or feature alignment.")

st.markdown("---")


