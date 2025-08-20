import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

# ----------------------------------------
# Streamlit App Title
# ----------------------------------------
st.set_page_config(page_title="Loan Default Risk Prediction", layout="wide")
st.title("Loan Default Risk Prediction App")
st.markdown("""
Upload dataset, train models, visualize metrics, predict new applicants (single or batch),
highlight high-risk applicants, flag top-N risky cases, and explore risk distribution.
""")

# ----------------------------------------
# Upload Dataset
# ----------------------------------------
uploaded_file = st.file_uploader("Upload CSV file with TARGET column", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())

    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df = pd.get_dummies(df, drop_first=True)

    if 'TARGET' not in df.columns:
        st.error("Dataset must contain 'TARGET' column.")
    else:
        X = df.drop('TARGET', axis=1)
        y = df['TARGET']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ----------------------------------------
        # Model Selection
        # ----------------------------------------
        model_choice = st.selectbox("Select Model(s) to Train", 
                                    ("Logistic Regression", "CatBoost", "Both"))

        if st.button("Train Model(s)"):
            results = {}
            auc_scores = {}

            # Logistic Regression
            if model_choice in ("Logistic Regression", "Both"):
                log_model = LogisticRegression(max_iter=5000)
                log_model.fit(X_train_scaled, y_train)
                y_pred_prob_log = log_model.predict_proba(X_test_scaled)[:,1]
                results['Logistic Regression'] = y_pred_prob_log
                auc_scores['Logistic Regression'] = roc_auc_score(y_test, y_pred_prob_log)
                st.success("Logistic Regression Trained Successfully!")

            # CatBoost
            if model_choice in ("CatBoost", "Both"):
                cat_model = CatBoostClassifier(verbose=0)
                cat_model.fit(X_train, y_train)
                y_pred_prob_cat = cat_model.predict_proba(X_test)[:,1]
                results['CatBoost'] = y_pred_prob_cat
                auc_scores['CatBoost'] = roc_auc_score(y_test, y_pred_prob_cat)
                st.success("CatBoost Trained Successfully!")

            # ----------------------------------------
            # Business Cost Optimization
            # ----------------------------------------
            cost_fp = 1000
            cost_fn = 5000
            thresholds = np.linspace(0,1,101)
            optimal_thresholds = {}

            def total_cost(y_true, y_prob, threshold):
                y_pred = (y_prob >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                return fp*cost_fp + fn*cost_fn

            for model_name, y_prob in results.items():
                optimal_thresh = thresholds[np.argmin([total_cost(y_test, y_prob, t) for t in thresholds])]
                optimal_thresholds[model_name] = optimal_thresh
                st.write(f"Optimal Threshold - {model_name}: {optimal_thresh:.2f}")

            # Business cost comparison
            total_costs = [total_cost(y_test, y_prob, optimal_thresholds[model_name]) 
                           for model_name, y_prob in results.items()]
            st.write("Total Business Cost Comparison")
            st.bar_chart(pd.DataFrame({'Models': list(results.keys()),
                                       'Total Cost': total_costs}))

            # ----------------------------------------
            # ROC Curve Plot
            # ----------------------------------------
            st.subheader("ROC Curves")
            fig, ax = plt.subplots(figsize=(7,5))
            for model_name, y_prob in results.items():
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_scores[model_name]:.2f})")
            ax.plot([0,1],[0,1],'--', color='gray')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            st.pyplot(fig)

            # ----------------------------------------
            # Precision-Recall Curve Plot
            # ----------------------------------------
            st.subheader("Precision-Recall Curves")
            fig2, ax2 = plt.subplots(figsize=(7,5))
            for model_name, y_prob in results.items():
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                ax2.plot(recall, precision, label=f"{model_name}")
            ax2.set_xlabel("Recall")
            ax2.set_ylabel("Precision")
            ax2.set_title("Precision-Recall Curve")
            ax2.legend()
            st.pyplot(fig2)

            # ----------------------------------------
            # Predict New Applicant(s)
            # ----------------------------------------
            st.subheader("Predict Loan Default for New Applicant(s)")
            input_option = st.radio("Input Type:", ("Single Applicant", "Batch CSV Upload"))
            risk_threshold = st.slider("Set High-Risk Probability Threshold", 0.0, 1.0, 0.5, 0.01)
            top_n = st.number_input("Number of Top High-Risk Applicants to Flag", min_value=1, value=5, step=1)

            # Single Applicant Input
            if input_option == "Single Applicant":
                st.markdown("Enter feature values for a single applicant:")
                input_data = {}
                for col in X.columns[:10]:  # First 10 features for simplicity
                    input_data[col] = st.number_input(f"{col}", value=float(X[col].median()))
                if st.button("Predict for Applicant"):
                    new_app = pd.DataFrame([input_data])
                    new_app = new_app.reindex(columns=X.columns, fill_value=0)
                    predictions = {}
                    if 'Logistic Regression' in results:
                        new_scaled = scaler.transform(new_app)
                        prob_log = log_model.predict_proba(new_scaled)[0,1]
                        predictions['Logistic Regression'] = prob_log
                    if 'CatBoost' in results:
                        prob_cat = cat_model.predict_proba(new_app)[0,1]
                        predictions['CatBoost'] = prob_cat

                    # Highlight high risk
                    for model_name, prob in predictions.items():
                        if prob >= risk_threshold:
                            st.markdown(f"⚠️ **{model_name} Predicted Default Probability: {prob:.2f} (High Risk)**")
                        else:
                            st.write(f"{model_name} Predicted Default Probability: {prob:.2f}")

            # Batch Input
            else:
                uploaded_new = st.file_uploader("Upload CSV for new applicants", type=["csv"], key="newcsv")
                if uploaded_new is not None:
                    new_df = pd.read_csv(uploaded_new)
                    st.write("New Applicants Preview:", new_df.head())
                    new_df_aligned = pd.get_dummies(new_df)
                    new_df_aligned = new_df_aligned.reindex(columns=X.columns, fill_value=0)

                    if st.button("Predict Batch"):
                        preds = {}
                        if 'Logistic Regression' in results:
                            new_scaled = scaler.transform(new_df_aligned)
                            preds['Logistic Regression'] = log_model.predict_proba(new_scaled)[:,1]
                        if 'CatBoost' in results:
                            preds['CatBoost'] = cat_model.predict_proba(new_df_aligned)[:,1]

                        pred_df = pd.DataFrame(preds)

                        # Highlight high-risk
                        st.write("Predicted Default Probabilities (High-risk highlighted in red):")
                        def highlight_high_risk(val):
                            color = 'red' if val >= risk_threshold else ''
                            return f'background-color: {color}'
                        st.dataframe(pred_df.style.applymap(highlight_high_risk))

                        # Top-N High-Risk Applicants
                        st.subheader(f"Top {top_n} High-Risk Applicants")
                        for model_name in pred_df.columns:
                            st.write(f"Top {top_n} for {model_name}:")
                            top_risk = pred_df[model_name].sort_values(ascending=False).head(top_n)
                            st.dataframe(top_risk)

                        # High-Risk Visualization
                        st.subheader("High-Risk Applicants Visualization")
                        high_risk_counts = {}
                        for model_name in pred_df.columns:
                            high_risk_counts[model_name] = (pred_df[model_name] >= risk_threshold).sum()
                        st.bar_chart(pd.DataFrame({
                            'Model': list(high_risk_counts.keys()),
                            'High-Risk Count': list(high_risk_counts.values())
                        }).set_index('Model'))

                        # ----------------------------------------
                        # Interactive Scatter Plot
                        # ----------------------------------------
                        st.subheader("Scatter Plot: Predicted Risk vs Feature")
                        feature_options = X.select_dtypes(include=['number']).columns.tolist()
                        selected_feature = st.selectbox("Select Feature for X-axis", feature_options)

                        for model_name in pred_df.columns:
                            st.write(f"Scatter Plot for {model_name}:")
                            plt.figure(figsize=(8,5))
                            plt.scatter(new_df_aligned[selected_feature], pred_df[model_name],
                                        c=(pred_df[model_name] >= risk_threshold), cmap='coolwarm', alpha=0.6)
                            plt.colorbar(label=f'High-Risk (Red=True / Blue=False)')
                            plt.xlabel(selected_feature)
                            plt.ylabel("Predicted Default Probability")
                            plt.title(f"{model_name}: Risk vs {selected_feature}")
                            st.pyplot(plt)

                        # Download predictions
                        st.download_button("Download Predictions",
                                           pred_df.to_csv(index=False).encode('utf-8'),
                                           file_name="predictions.csv",
                                           mime="text/csv")

            # ----------------------------------------
            # Download Processed Dataset
            # ----------------------------------------
            st.download_button("Download Processed Dataset",
                               df.to_csv(index=False).encode('utf-8'),
                               file_name="processed_dataset.csv",
                               mime="text/csv")
