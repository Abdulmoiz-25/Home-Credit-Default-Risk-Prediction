import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)

# ---------------------------------------------------
# Streamlit page config MUST be first
# ---------------------------------------------------
st.set_page_config(page_title="Loan Default Risk Prediction", layout="wide")

# ---------------------------------------------------
# Load Pre-trained Models and Scaler
# ---------------------------------------------------
@st.cache_resource
def load_models():
    log_m = joblib.load("log_model.pkl")
    sc = joblib.load("scaler.pkl")
    cb = CatBoostClassifier()
    cb.load_model("cat_model.cbm")
    return log_m, sc, cb

log_model, scaler, cat_model = load_models()

# Load dataset (for feature alignment)
@st.cache_data
def load_sample():
    try:
        df = pd.read_csv("sample_train.csv")
        # Handle missing values (numeric columns) - fill with median
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Apply one-hot encoding exactly like training
        df_encoded = pd.get_dummies(df, drop_first=True)
        X = df_encoded.drop('TARGET', axis=1)
        return X
    except FileNotFoundError:
        # Create sample_train.csv with actual Home Credit structure
        sample_data = {
            'SK_ID_CURR': [100002, 100003, 100004, 100006, 100007],
            'NAME_CONTRACT_TYPE': ['Cash loans', 'Cash loans', 'Revolving loans', 'Cash loans', 'Cash loans'],
            'CODE_GENDER': ['M', 'F', 'M', 'F', 'M'],
            'FLAG_OWN_CAR': ['N', 'N', 'Y', 'N', 'N'],
            'FLAG_OWN_REALTY': ['Y', 'N', 'Y', 'Y', 'Y'],
            'CNT_CHILDREN': [0, 0, 0, 0, 0],
            'AMT_INCOME_TOTAL': [202500, 270000, 67500, 135000, 121500],
            'AMT_CREDIT': [406597.5, 1293502.5, 135000, 312682.5, 513000],
            'AMT_ANNUITY': [24700.5, 35698.5, 6750, 29686.5, 21865.5],
            'AMT_GOODS_PRICE': [351000, 1129500, 135000, 297000, 513000],
            'NAME_TYPE_SUITE': ['Unaccompanied', 'Family', 'Unaccompanied', 'Unaccompanied', 'Unaccompanied'],
            'NAME_INCOME_TYPE': ['Working', 'State servant', 'Working', 'Working', 'Working'],
            'NAME_EDUCATION_TYPE': ['Secondary / secondary special', 'Higher education', 'Secondary / secondary special', 'Secondary / secondary special', 'Secondary / secondary special'],
            'NAME_FAMILY_STATUS': ['Single / not married', 'Married', 'Single / not married', 'Civil marriage', 'Single / not married'],
            'NAME_HOUSING_TYPE': ['House / apartment', 'House / apartment', 'House / apartment', 'House / apartment', 'House / apartment'],
            'REGION_POPULATION_RELATIVE': [0.018801, 0.003541, 0.010032, 0.008019, 0.028663],
            'DAYS_BIRTH': [-9461, -16765, -19046, -19005, -19932],
            'DAYS_EMPLOYED': [-637, -1188, -225, -3039, -3038],
            'DAYS_REGISTRATION': [-3648, -1186, -4260, -9833, -4311],
            'DAYS_ID_PUBLISH': [-2120, -291, -2531, -2437, -3458],
            'OWN_CAR_AGE': [None, None, 26, None, None],
            'FLAG_MOBIL': [1, 1, 1, 1, 1],
            'FLAG_EMP_PHONE': [1, 1, 1, 1, 1],
            'FLAG_WORK_PHONE': [0, 0, 1, 0, 0],
            'FLAG_CONT_MOBILE': [1, 1, 1, 1, 1],
            'FLAG_PHONE': [1, 1, 1, 0, 0],
            'FLAG_EMAIL': [0, 0, 0, 0, 0],
            'OCCUPATION_TYPE': ['Laborers', 'Core staff', 'Laborers', 'Laborers', 'Core staff'],
            'CNT_FAM_MEMBERS': [1, 2, 1, 2, 1],
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
            'EXT_SOURCE_1': [0.083037, 0.311267, None, None, None],
            'EXT_SOURCE_2': [0.262949, 0.622246, 0.555912, 0.650442, 0.322738],
            'EXT_SOURCE_3': [0.139376, None, 0.729567, None, None],
            'TARGET': [1, 0, 0, 0, 0]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv("sample_train.csv", index=False)
        
        # Handle missing values (numeric columns)
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Apply one-hot encoding like in training
        df_encoded = pd.get_dummies(df, drop_first=True)
        X = df_encoded.drop("TARGET", axis=1)
        return X

X = load_sample()

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.title("Loan Default Risk Prediction App")
st.markdown("""
Predict loan default risk using pre-trained models (**Logistic Regression** & **CatBoost**).  
Visualize metrics, compare business costs, highlight high-risk applicants, inspect confusion matrices, and download results.
""")

# Sidebar
st.sidebar.header("App Settings")
mode = st.sidebar.radio("Select Mode:", ["Single Applicant", "Batch Prediction"])
risk_threshold = st.sidebar.slider("High-Risk Threshold (for decisions & visuals)", 0.0, 1.0, 0.5, 0.01)
top_n = st.sidebar.number_input("Top-N High-Risk Applicants", min_value=1, value=5, step=1)
selected_models = st.sidebar.multiselect(
    "Select Models",
    ["Logistic Regression", "CatBoost"],
    default=["Logistic Regression", "CatBoost"]
)

# ---------------------------------------------------
# Business Cost Settings
# ---------------------------------------------------
cost_fp = 1000
cost_fn = 5000

def total_cost(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp * cost_fp + fn * cost_fn

def plot_cm(ax, cm, title="Confusion Matrix"):
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_xticklabels(["No Default","Default"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["No Default","Default"])

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return im

def cm_counts_to_df(cm):
    tn, fp, fn, tp = cm.ravel()
    return pd.DataFrame({
        "TN":[tn], "FP":[fp], "FN":[fn], "TP":[tp]
    })

# ---------------------------------------------------
# ---------------------------------------------------
training_feature_names = X.columns.tolist()

# ---------------------------------------------------
# Single Applicant Prediction
# ---------------------------------------------------
if mode == "Single Applicant":
    st.subheader("Single Applicant Prediction")

    st.write("Enter applicant details:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amt_income = st.number_input("Annual Income (AMT_INCOME_TOTAL)", value=202500, min_value=0)
        amt_credit = st.number_input("Credit Amount (AMT_CREDIT)", value=406597, min_value=0)
        amt_annuity = st.number_input("Loan Annuity (AMT_ANNUITY)", value=24700, min_value=0)
        amt_goods_price = st.number_input("Goods Price (AMT_GOODS_PRICE)", value=351000, min_value=0)
    
    with col2:
        code_gender = st.selectbox("Gender", ["M", "F"])
        name_contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
        flag_own_car = st.selectbox("Owns Car", ["Y", "N"])
        flag_own_realty = st.selectbox("Owns Realty", ["Y", "N"])
    
    col3, col4 = st.columns(2)
    
    with col3:
        cnt_children = st.number_input("Number of Children", value=0, min_value=0, max_value=20)
        days_birth = st.number_input("Age (years)", value=26, min_value=18, max_value=100)
        days_employed = st.number_input("Years Employed", value=2, min_value=0, max_value=50)
    
    with col4:
        name_income_type = st.selectbox("Income Type", ["Working", "State servant", "Commercial associate", "Pensioner"])
        name_education_type = st.selectbox("Education", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
        name_family_status = st.selectbox("Family Status", ["Single / not married", "Married", "Civil marriage", "Separated", "Widow"])
    
    predict_button = st.button("🔮 Predict Default Risk", type="primary")
    
    if predict_button:
        new_app_aligned = pd.DataFrame(0, index=[0], columns=X.columns)
        
        # Map user inputs to actual training feature names
        feature_mapping = {
            'SK_ID_CURR': 100000,  # Dummy ID
            'AMT_INCOME_TOTAL': amt_income,
            'AMT_CREDIT': amt_credit,
            'AMT_ANNUITY': amt_annuity,
            'AMT_GOODS_PRICE': amt_goods_price,
            'CNT_CHILDREN': cnt_children,
            'DAYS_BIRTH': -days_birth * 365,  # Convert to negative days
            'DAYS_EMPLOYED': -days_employed * 365,  # Convert to negative days
            'FLAG_MOBIL': 1,
            'FLAG_EMP_PHONE': 1,
            'FLAG_CONT_MOBILE': 1,
            'CNT_FAM_MEMBERS': cnt_children + 1,
            'REGION_RATING_CLIENT': 2,
            'REGION_RATING_CLIENT_W_CITY': 2,
            'HOUR_APPR_PROCESS_START': 10,
        }
        
        # Fill in the mapped features
        for feature, value in feature_mapping.items():
            if feature in new_app_aligned.columns:
                new_app_aligned[feature] = value
        
        # Handle categorical features (one-hot encoded)
        categorical_mappings = {
            'CODE_GENDER_M': 1 if code_gender == 'M' else 0,
            'NAME_CONTRACT_TYPE_Revolving loans': 1 if name_contract_type == 'Revolving loans' else 0,
            'FLAG_OWN_CAR_Y': 1 if flag_own_car == 'Y' else 0,
            'FLAG_OWN_REALTY_Y': 1 if flag_own_realty == 'Y' else 0,
            'NAME_INCOME_TYPE_Working': 1 if name_income_type == 'Working' else 0,
            'NAME_INCOME_TYPE_State servant': 1 if name_income_type == 'State servant' else 0,
            'NAME_INCOME_TYPE_Commercial associate': 1 if name_income_type == 'Commercial associate' else 0,
            'NAME_INCOME_TYPE_Pensioner': 1 if name_income_type == 'Pensioner' else 0,
            'NAME_EDUCATION_TYPE_Secondary / secondary special': 1 if name_education_type == 'Secondary / secondary special' else 0,
            'NAME_EDUCATION_TYPE_Higher education': 1 if name_education_type == 'Higher education' else 0,
            'NAME_EDUCATION_TYPE_Incomplete higher': 1 if name_education_type == 'Incomplete higher' else 0,
            'NAME_EDUCATION_TYPE_Lower secondary': 1 if name_education_type == 'Lower secondary' else 0,
            'NAME_FAMILY_STATUS_Single / not married': 1 if name_family_status == 'Single / not married' else 0,
            'NAME_FAMILY_STATUS_Married': 1 if name_family_status == 'Married' else 0,
            'NAME_FAMILY_STATUS_Civil marriage': 1 if name_family_status == 'Civil marriage' else 0,
            'NAME_FAMILY_STATUS_Separated': 1 if name_family_status == 'Separated' else 0,
            'NAME_FAMILY_STATUS_Widow': 1 if name_family_status == 'Widow' else 0,
        }
        
        # Fill in categorical features
        for feature, value in categorical_mappings.items():
            if feature in new_app_aligned.columns:
                new_app_aligned[feature] = value

        st.write(f"Debug: DataFrame shape: {new_app_aligned.shape}")
        st.write(f"Debug: Expected features: {len(X.columns)}")
        
        preds = {}
        if "Logistic Regression" in selected_models:
            try:
                scaled = scaler.transform(new_app_aligned)
                preds["Logistic Regression"] = float(log_model.predict_proba(scaled)[:, 1][0])
            except Exception as e:
                st.error(f"Error with Logistic Regression prediction: {str(e)}")
                st.info("Feature alignment issue. Check that the scaler was trained on the same features.")

        if "CatBoost" in selected_models:
            try:
                preds["CatBoost"] = float(cat_model.predict_proba(new_app_aligned)[:, 1][0])
            except Exception as e:
                st.error(f"Error with CatBoost prediction: {str(e)}")
                st.info("Feature alignment issue. Check that CatBoost was trained on the same features.")

        if preds:
            st.write("### Prediction Results")
            for model, prob in preds.items():
                result = "⚠️ High Risk" if prob >= risk_threshold else "✅ Low Risk"
                st.write(f"**{model}** → Probability: {prob:.3f} → {result}")
        else:
            st.warning("No predictions could be generated. Please check the model files and feature alignment.")

# ---------------------------------------------------
# Batch Prediction
# ---------------------------------------------------
else:
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV file with applicants' data", type=["csv"])

    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)

        has_target = "TARGET" in new_df.columns
        if has_target:
            y_true = new_df["TARGET"].astype(int)
            feats = new_df.drop("TARGET", axis=1)
        else:
            feats = new_df.copy()

        # Handle missing values (numeric columns)
        numeric_cols = feats.select_dtypes(include=['number']).columns
        feats[numeric_cols] = feats[numeric_cols].fillna(feats[numeric_cols].median())
        
        # Apply one-hot encoding
        feats_encoded = pd.get_dummies(feats, drop_first=True)
        
        # Align with training features
        new_df_aligned = feats_encoded.reindex(columns=training_feature_names, fill_value=0)

        # Predictions
        preds = {}
        if "Logistic Regression" in selected_models:
            new_scaled = scaler.transform(new_df_aligned)
            preds["Logistic Regression"] = log_model.predict_proba(new_scaled)[:, 1]

        if "CatBoost" in selected_models:
            preds["CatBoost"] = cat_model.predict_proba(new_df_aligned)[:, 1]

        pred_df = pd.DataFrame(preds, index=new_df.index)
        if has_target:
            pred_df["TARGET"] = y_true.values

        st.write("### Predictions")
        st.dataframe(pred_df)

        # Highlight high-risk
        st.write("### High-Risk Applicants (Highlighted)")
        def highlight_high_risk(val):
            color = 'red' if (isinstance(val, (float, int)) and val >= risk_threshold) else ''
            return f'background-color: {color}'
        if len(preds) > 0:
            st.dataframe(pred_df.style.applymap(highlight_high_risk, subset=list(preds.keys())))

        # Top-N risky applicants
        st.subheader(f"Top {top_n} High-Risk Applicants")
        for model_name in pred_df.columns.drop("TARGET", errors="ignore"):
            st.write(f"Top {top_n} for **{model_name}**:")
            top_risk = pred_df[model_name].sort_values(ascending=False).head(top_n)
            st.dataframe(top_risk)

        # High-risk counts
        st.subheader("High-Risk Applicants Count")
        high_risk_counts = {m: int((pred_df[m] >= risk_threshold).sum()) for m in pred_df.columns if m != "TARGET"}
        if high_risk_counts:
            st.bar_chart(pd.DataFrame({
                'Model': list(high_risk_counts.keys()),
                'High-Risk Count': list(high_risk_counts.values())
            }).set_index('Model'))

        # Scatter Plot
        st.subheader("Scatter Plot: Predicted Risk vs Feature")
        feature_options = X.select_dtypes(include=['number']).columns.tolist()
        if len(feature_options) == 0:
            feature_options = list(X.columns)
        selected_feature = st.selectbox("Select Feature for X-axis", feature_options)
        for model_name in pred_df.columns.drop("TARGET", errors="ignore"):
            st.write(f"Scatter Plot for **{model_name}**:")
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(
                new_df_aligned[selected_feature],
                pred_df[model_name],
                c=(pred_df[model_name] >= risk_threshold),
                cmap='coolwarm', alpha=0.6
            )
            plt.colorbar(scatter, ax=ax, label='High-Risk (Red=True / Blue=False)')
            ax.set_xlabel(selected_feature)
            ax.set_ylabel("Predicted Default Probability")
            ax.set_title(f"{model_name}: Risk vs {selected_feature}")
            st.pyplot(fig)

        # ------------------------------------
        # Evaluation: ROC, PR, Cost, Confusion Matrices
        # ------------------------------------
        if has_target and len(preds) > 0:
            st.subheader("Model Evaluation (with Ground Truth)")

            auc_scores = {}
            thresholds = np.linspace(0, 1, 101)
            optimal_thresholds = {}
            total_costs_at_opt = {}

            # Compute AUC + optimal threshold via business cost
            for model_name, y_prob in preds.items():
                auc_scores[model_name] = roc_auc_score(y_true, y_prob)
                costs = [total_cost(y_true, y_prob, t) for t in thresholds]
                optimal_thresh = float(thresholds[int(np.argmin(costs))])
                optimal_thresholds[model_name] = optimal_thresh
                total_costs_at_opt[model_name] = total_cost(y_true, y_prob, optimal_thresh)

            # Show optimal thresholds
            st.write("**Optimal Thresholds (minimizing FP*{:,} + FN*{:,})**".format(cost_fp, cost_fn))
            opt_table = pd.DataFrame({
                "Model": list(optimal_thresholds.keys()),
                "Optimal Threshold": [round(optimal_thresholds[m], 3) for m in optimal_thresholds],
                "AUC": [round(auc_scores[m], 3) for m in optimal_thresholds],
                "Total Cost @ Optimal": [total_costs_at_opt[m] for m in optimal_thresholds],
            })
            st.dataframe(opt_table.set_index("Model"))

            # Business cost comparison bar
            st.write("### Business Cost Comparison (@ each model's optimal threshold)")
            st.bar_chart(pd.DataFrame({
                "Model": list(total_costs_at_opt.keys()),
                "Total Cost": list(total_costs_at_opt.values())
            }).set_index("Model"))

            # ROC Curves
            st.subheader("ROC Curves")
            fig, ax = plt.subplots(figsize=(7, 5))
            for model_name, y_prob in preds.items():
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_scores[model_name]:.3f})")
            ax.plot([0, 1], [0, 1], '--', color='gray')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            st.pyplot(fig)

            # Precision-Recall Curves
            st.subheader("Precision-Recall Curves")
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            for model_name, y_prob in preds.items():
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                ax2.plot(recall, precision, label=f"{model_name}")
            ax2.set_xlabel("Recall")
            ax2.set_ylabel("Precision")
            ax2.set_title("Precision-Recall Curve")
            ax2.legend()
            st.pyplot(fig2)

            # Confusion Matrices at:
            # 1) Sidebar threshold
            # 2) Model's optimal threshold
            st.subheader("Confusion Matrices")

            for model_name, y_prob in preds.items():
                st.markdown(f"#### {model_name}")

                # --- At Sidebar Threshold ---
                y_pred_sidebar = (y_prob >= risk_threshold).astype(int)
                cm_sidebar = confusion_matrix(y_true, y_pred_sidebar)
                c1, c2 = st.columns([1.2, 1])
                with c1:
                    fig_cm1, ax_cm1 = plt.subplots(figsize=(4.5, 4))
                    plot_cm(ax_cm1, cm_sidebar, title=f"Threshold = {risk_threshold:.2f}")
                    st.pyplot(fig_cm1)
                with c2:
                    st.write("Counts @ Sidebar Threshold")
                    st.dataframe(cm_counts_to_df(cm_sidebar))

                # --- At Optimal Threshold ---
                opt_t = optimal_thresholds[model_name]
                y_pred_opt = (y_prob >= opt_t).astype(int)
                cm_opt = confusion_matrix(y_true, y_pred_opt)
                c3, c4 = st.columns([1.2, 1])
                with c3:
                    fig_cm2, ax_cm2 = plt.subplots(figsize=(4.5, 4))
                    plot_cm(ax_cm2, cm_opt, title=f"Optimal Threshold = {opt_t:.2f}")
                    st.pyplot(fig_cm2)
                with c4:
                    st.write("Counts @ Optimal Threshold")
                    st.dataframe(cm_counts_to_df(cm_opt))

        # Download Predictions
        st.download_button(
            "Download Predictions (CSV)",
            pred_df.to_csv(index=False).encode('utf-8'),
            file_name="predictions.csv",
            mime="text/csv"
        )

st.success("✅ App is ready. Upload data or enter applicant details to get predictions.")
