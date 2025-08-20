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
        X = df.drop("TARGET", axis=1)
        return X
    except FileNotFoundError:
        # The models expect features like AMT_ANNUITY, AMT_CREDIT, etc.
        feature_names = [
            'SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
            'NAME_CONTRACT_TYPE_Cash loans', 'NAME_CONTRACT_TYPE_Revolving loans',
            'CODE_GENDER_F', 'CODE_GENDER_M', 'FLAG_OWN_CAR_N', 'FLAG_OWN_CAR_Y',
            'FLAG_OWN_REALTY_N', 'FLAG_OWN_REALTY_Y', 'CNT_CHILDREN', 'AMT_REQ_CREDIT_BUREAU_HOUR',
            'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
            'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'NAME_TYPE_SUITE_Children',
            'NAME_TYPE_SUITE_Family', 'NAME_TYPE_SUITE_Group of people', 'NAME_TYPE_SUITE_Other_A',
            'NAME_TYPE_SUITE_Other_B', 'NAME_TYPE_SUITE_Spouse, partner', 'NAME_TYPE_SUITE_Unaccompanied',
            'NAME_INCOME_TYPE_Businessman', 'NAME_INCOME_TYPE_Commercial associate', 
            'NAME_INCOME_TYPE_Maternity leave', 'NAME_INCOME_TYPE_Pensioner', 'NAME_INCOME_TYPE_State servant',
            'NAME_INCOME_TYPE_Student', 'NAME_INCOME_TYPE_Unemployed', 'NAME_INCOME_TYPE_Working',
            'NAME_EDUCATION_TYPE_Academic degree', 'NAME_EDUCATION_TYPE_Higher education',
            'NAME_EDUCATION_TYPE_Incomplete higher', 'NAME_EDUCATION_TYPE_Lower secondary',
            'NAME_EDUCATION_TYPE_Secondary / secondary special', 'NAME_FAMILY_STATUS_Civil marriage',
            'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Separated', 'NAME_FAMILY_STATUS_Single / not married',
            'NAME_FAMILY_STATUS_Unknown', 'NAME_FAMILY_STATUS_Widow', 'NAME_HOUSING_TYPE_Co-op apartment',
            'NAME_HOUSING_TYPE_House / apartment', 'NAME_HOUSING_TYPE_Municipal apartment',
            'NAME_HOUSING_TYPE_Office apartment', 'NAME_HOUSING_TYPE_Rented apartment',
            'NAME_HOUSING_TYPE_With parents', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
            'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
            'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
            'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE_Accountants', 'OCCUPATION_TYPE_Cleaning staff',
            'OCCUPATION_TYPE_Cooking staff', 'OCCUPATION_TYPE_Core staff', 'OCCUPATION_TYPE_Drivers',
            'OCCUPATION_TYPE_HR staff', 'OCCUPATION_TYPE_High skill tech staff', 'OCCUPATION_TYPE_IT staff',
            'OCCUPATION_TYPE_Laborers', 'OCCUPATION_TYPE_Low-skill Laborers', 'OCCUPATION_TYPE_Managers',
            'OCCUPATION_TYPE_Medicine staff', 'OCCUPATION_TYPE_Private service staff',
            'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Sales staff', 'OCCUPATION_TYPE_Secretaries',
            'OCCUPATION_TYPE_Security staff', 'OCCUPATION_TYPE_Waiters/barmen staff', 'CNT_FAM_MEMBERS',
            'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START_FRIDAY',
            'WEEKDAY_APPR_PROCESS_START_MONDAY', 'WEEKDAY_APPR_PROCESS_START_SATURDAY',
            'WEEKDAY_APPR_PROCESS_START_SUNDAY', 'WEEKDAY_APPR_PROCESS_START_THURSDAY',
            'WEEKDAY_APPR_PROCESS_START_TUESDAY', 'WEEKDAY_APPR_PROCESS_START_WEDNESDAY',
            'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
            'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
            'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE_Advertising', 'ORGANIZATION_TYPE_Agriculture',
            'ORGANIZATION_TYPE_Bank', 'ORGANIZATION_TYPE_Business Entity Type 1',
            'ORGANIZATION_TYPE_Business Entity Type 2', 'ORGANIZATION_TYPE_Business Entity Type 3',
            'ORGANIZATION_TYPE_Cleaning', 'ORGANIZATION_TYPE_Construction', 'ORGANIZATION_TYPE_Culture',
            'ORGANIZATION_TYPE_Electricity', 'ORGANIZATION_TYPE_Emergency', 'ORGANIZATION_TYPE_Government',
            'ORGANIZATION_TYPE_Hotel', 'ORGANIZATION_TYPE_Housing', 'ORGANIZATION_TYPE_Industry: type 1',
            'ORGANIZATION_TYPE_Industry: type 10', 'ORGANIZATION_TYPE_Industry: type 11',
            'ORGANIZATION_TYPE_Industry: type 12', 'ORGANIZATION_TYPE_Industry: type 13',
            'ORGANIZATION_TYPE_Industry: type 2', 'ORGANIZATION_TYPE_Industry: type 3',
            'ORGANIZATION_TYPE_Industry: type 4', 'ORGANIZATION_TYPE_Industry: type 5',
            'ORGANIZATION_TYPE_Industry: type 6', 'ORGANIZATION_TYPE_Industry: type 7',
            'ORGANIZATION_TYPE_Industry: type 8', 'ORGANIZATION_TYPE_Industry: type 9',
            'ORGANIZATION_TYPE_Insurance', 'ORGANIZATION_TYPE_Kindergarten', 'ORGANIZATION_TYPE_Legal Services',
            'ORGANIZATION_TYPE_Medicine', 'ORGANIZATION_TYPE_Military', 'ORGANIZATION_TYPE_Mobile',
            'ORGANIZATION_TYPE_Other', 'ORGANIZATION_TYPE_Police', 'ORGANIZATION_TYPE_Postal',
            'ORGANIZATION_TYPE_Realtor', 'ORGANIZATION_TYPE_Religion', 'ORGANIZATION_TYPE_Restaurant',
            'ORGANIZATION_TYPE_School', 'ORGANIZATION_TYPE_Security', 'ORGANIZATION_TYPE_Security Ministries',
            'ORGANIZATION_TYPE_Self-employed', 'ORGANIZATION_TYPE_Services', 'ORGANIZATION_TYPE_Telecom',
            'ORGANIZATION_TYPE_Trade: type 1', 'ORGANIZATION_TYPE_Trade: type 2', 'ORGANIZATION_TYPE_Trade: type 3',
            'ORGANIZATION_TYPE_Trade: type 4', 'ORGANIZATION_TYPE_Trade: type 5', 'ORGANIZATION_TYPE_Trade: type 6',
            'ORGANIZATION_TYPE_Trade: type 7', 'ORGANIZATION_TYPE_Transport: type 1',
            'ORGANIZATION_TYPE_Transport: type 2', 'ORGANIZATION_TYPE_Transport: type 3',
            'ORGANIZATION_TYPE_Transport: type 4', 'ORGANIZATION_TYPE_University', 'ORGANIZATION_TYPE_XNA',
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG',
            'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG',
            'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
            'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE',
            'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE',
            'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE',
            'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
            'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
            'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI',
            'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI',
            'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE_not specified', 'FONDKAPREMONT_MODE_org spec account',
            'FONDKAPREMONT_MODE_reg oper account', 'FONDKAPREMONT_MODE_reg oper spec account',
            'HOUSETYPE_MODE_block of flats', 'HOUSETYPE_MODE_specific housing', 'HOUSETYPE_MODE_terraced house',
            'TOTALAREA_MODE', 'WALLSMATERIAL_MODE_Block', 'WALLSMATERIAL_MODE_Mixed',
            'WALLSMATERIAL_MODE_Monolithic', 'WALLSMATERIAL_MODE_Others', 'WALLSMATERIAL_MODE_Panel',
            'WALLSMATERIAL_MODE_Stone, brick', 'WALLSMATERIAL_MODE_Wooden', 'EMERGENCYSTATE_MODE_No',
            'EMERGENCYSTATE_MODE_Yes', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
            'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE',
            'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5',
            'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
            'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13',
            'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17',
            'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'
        ]
        
        # Create dummy DataFrame with all expected features
        dummy_data = {col: [0] for col in feature_names}
        return pd.DataFrame(dummy_data)

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
# Single Applicant Prediction
# ---------------------------------------------------
if mode == "Single Applicant":
    st.subheader("Single Applicant Prediction")

    st.write("Enter applicant details:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input("Annual Income", value=30000, min_value=0)
        age = st.number_input("Age", value=28, min_value=18, max_value=100)
    
    with col2:
        loan_amount = st.number_input("Loan Amount", value=5000, min_value=0)
        credit_score = st.number_input("Credit Score", value=650, min_value=300, max_value=850)
    
    predict_button = st.button("🔮 Predict Default Risk", type="primary")
    
    if predict_button:
        input_data = {
            'income': income,
            'age': age,
            'loan_amount': loan_amount,
            'credit_score': credit_score
        }
        
        new_app_df = pd.DataFrame([{col: 0 for col in X.columns}])
        
        # Map user inputs to expected feature names
        new_app_df['SK_ID_CURR'] = 1  # Dummy ID
        new_app_df['AMT_INCOME_TOTAL'] = income
        new_app_df['AMT_CREDIT'] = loan_amount
        new_app_df['AMT_ANNUITY'] = loan_amount * 0.1  # Estimate annuity as 10% of loan
        new_app_df['AMT_GOODS_PRICE'] = loan_amount
        new_app_df['DAYS_BIRTH'] = -age * 365  # Convert age to negative days
        new_app_df['CODE_GENDER_M'] = 1  # Default to male
        new_app_df['NAME_CONTRACT_TYPE_Cash loans'] = 1  # Default to cash loans
        new_app_df['FLAG_OWN_CAR_N'] = 1  # Default to no car
        new_app_df['FLAG_OWN_REALTY_Y'] = 1  # Default to owns realty

        preds = {}
        if "Logistic Regression" in selected_models:
            try:
                scaled = scaler.transform(new_app_df.values)
                preds["Logistic Regression"] = float(log_model.predict_proba(scaled)[:, 1][0])
            except Exception as e:
                st.error(f"Error with Logistic Regression prediction: {str(e)}")
                st.info("Feature alignment issue. Check that the scaler was trained on the same features.")

        if "CatBoost" in selected_models:
            try:
                preds["CatBoost"] = float(cat_model.predict_proba(new_app_df.values)[:, 1][0])
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

        new_df_aligned = feats.reindex(columns=X.columns, fill_value=0)

        # Predictions
        preds = {}
        if "Logistic Regression" in selected_models:
            new_scaled = scaler.transform(new_df_aligned.values)
            preds["Logistic Regression"] = log_model.predict_proba(new_scaled)[:, 1]

        if "CatBoost" in selected_models:
            preds["CatBoost"] = cat_model.predict_proba(new_df_aligned.values)[:, 1]

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
