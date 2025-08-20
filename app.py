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
    df = pd.read_csv("sample_train.csv")
    X_base = df.drop("TARGET", axis=1)
    return X_base

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

# Feature mapping from user-friendly names to actual training feature names
FEATURE_MAPPING = {
    'income': 'AMT_INCOME_TOTAL',
    'age': 'DAYS_BIRTH',  # Note: this will need to be converted (age * -365)
    'loan_amount': 'AMT_CREDIT',
    'credit_score': 'AMT_ANNUITY',  # Using as proxy - adjust as needed
    'employment_years': 'DAYS_EMPLOYED',  # Will need conversion
    'family_size': 'CNT_FAM_MEMBERS',
    'children_count': 'CNT_CHILDREN',
    'goods_price': 'AMT_GOODS_PRICE',
    'annuity': 'AMT_ANNUITY',
    'credit_bureau_requests': 'AMT_REQ_CREDIT_BUREAU_DAY'
}

# Reverse mapping for display
DISPLAY_MAPPING = {v: k for k, v in FEATURE_MAPPING.items()}

# ---------------------------------------------------
# Single Applicant Prediction
# ---------------------------------------------------
if mode == "Single Applicant":
    st.subheader("Single Applicant Prediction")

    applicant_data = {}
    
    # Define user-friendly inputs
    user_inputs = {
        'income': st.number_input("Annual Income", value=50000.0, min_value=0.0),
        'age': st.number_input("Age (years)", value=35, min_value=18, max_value=100),
        'loan_amount': st.number_input("Loan Amount", value=100000.0, min_value=0.0),
        'credit_score': st.number_input("Credit Score (as annuity proxy)", value=15000.0, min_value=0.0),
        'employment_years': st.number_input("Employment Years", value=5, min_value=0, max_value=50),
        'family_size': st.number_input("Family Size", value=2, min_value=1, max_value=10),
        'children_count': st.number_input("Number of Children", value=0, min_value=0, max_value=10),
        'goods_price': st.number_input("Goods Price", value=90000.0, min_value=0.0)
    }
    
    mapped_data = {}
    for user_key, value in user_inputs.items():
        if user_key in FEATURE_MAPPING:
            training_feature = FEATURE_MAPPING[user_key]
            
            # Handle special conversions
            if user_key == 'age':
                # Convert age to DAYS_BIRTH (negative days from birth)
                mapped_data[training_feature] = -value * 365
            elif user_key == 'employment_years':
                # Convert years to DAYS_EMPLOYED (negative days)
                mapped_data[training_feature] = -value * 365
            else:
                mapped_data[training_feature] = value

    # Build input row aligned to all features with proper defaults
    new_app = pd.DataFrame([mapped_data]).reindex(columns=X.columns, fill_value=0)

    preds = {}
    if "Logistic Regression" in selected_models:
        try:
            scaled = scaler.transform(new_app)
            preds["Logistic Regression"] = float(log_model.predict_proba(scaled)[:, 1][0])
        except Exception as e:
            st.error(f"Error with Logistic Regression prediction: {str(e)}")
            st.info("This usually means the scaler was trained on different features. Try using only the CatBoost model.")

    if "CatBoost" in selected_models:
        try:
            preds["CatBoost"] = float(cat_model.predict_proba(new_app)[:, 1][0])
        except Exception as e:
            st.error(f"Error with CatBoost prediction: {str(e)}")
            st.info("CatBoost model may have been trained on different features.")

    st.write("### Prediction Results")
    for model, prob in preds.items():
        result = "⚠️ High Risk" if prob >= risk_threshold else "✅ Low Risk"
        st.write(f"**{model}** → Probability: {prob:.3f} → {result}")

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

        # Align to training feature space
        new_df_aligned = feats.reindex(columns=X.columns, fill_value=0)

        # Predictions
        preds = {}
        if "Logistic Regression" in selected_models:
            new_scaled = scaler.transform(new_df_aligned.values)
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
