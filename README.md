# 📘 DeveloperHub Task 9 – Home Credit Default Risk Prediction App

## 📌 Project Objective  
Predict the probability of **loan default risk** for new applicants using **machine learning models**. The interactive dashboard leverages **Logistic Regression and CatBoost** for accurate prediction, with visually appealing results including probability cards, circular gauges, and risk badges.

---

## 📁 Dataset  
- **Name**: Home Credit Default Risk Dataset  
- **Source**: [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)  
- **Features include**:  
  - Personal information: Age, gender, education, family status, housing type  
  - Financial information: Income, credit, annuity, goods price  
  - Employment and registration: Days employed, days birth, ID publish days  
  - External sources: EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3  
  - Derived features: Averages, modes, medians of living areas, apartments, and other metrics  
  - Flags: Phone, email, mobile, document verification  
- **Frequency**: Single snapshot per applicant  

---

## 🛠️ Tools & Libraries Used  
- **Streamlit** – interactive web dashboard deployment  
- **Pandas** – data manipulation and preprocessing  
- **NumPy** – numerical operations  
- **CatBoost** – gradient boosting classifier for default risk  
- **Scikit-learn** – Logistic Regression, scaling, and preprocessing  
- **Matplotlib** – optional visualization (used internally)  
- **Joblib** – model serialization and caching  
- **JSON** – prediction download functionality  

---

## 🚀 Approach  

### 🔍 1. Data Loading & Preprocessing  
- Load training dataset for feature alignment and median calculation  
- Handle missing numerical values by filling with training medians  
- Encode categorical variables using one-hot encoding  
- Align new applicant features with training dataset columns  

### 🤖 2. Multi-Model Prediction  
- **Logistic Regression**: Probability of default after scaling  
- **CatBoost**: Tree-based gradient boosting classifier  
- Automatic handling of missing values for both models  
- Error handling with feedback in Streamlit interface  

### 📈 3. Advanced Visualization  
- Circular gauge to show average default probability  
- Colored probability badges: Very Low → Low → Moderate → High risk  
- Gradient-filled probability bars for intuitive reading  
- Responsive prediction cards per model  

### 💡 4. Interactive Dashboard  
- **Model selection**: Logistic Regression, CatBoost, or both  
- Real-time prediction on new applicant inputs  
- Download predictions in JSON format  
- Visual cues for quick loan approval decision  

---

## 📊 Key Features  

### 🎯 Prediction Capabilities  
- **Real-time risk assessment**: Single applicant predictions  
- **Multi-model support**: Logistic Regression and CatBoost  
- **Probability badges**: Visual risk level indicators  
- **Circular gauge**: Average risk visualization  

### 🔧 Technical Robustness  
- **Missing value handling**: Automatic filling with median values  
- **Model caching**: Preloaded models for fast predictions  
- **Error handling**: Streamlit feedback for prediction issues  
- **Feature alignment**: Automatic reindexing of new applicant features  

### 🎨 User Experience  
- **Dark theme**: Professional dashboard styling  
- **Interactive controls**: Form input for applicant information  
- **Visual clarity**: Clean cards and gradient bars for probabilities  
- **Download option**: JSON export of predicted probabilities  

---

## 📊 Results & Performance  
- **Logistic Regression**: Quick baseline probability predictions  
- **CatBoost**: More accurate tree-based model for complex patterns  
- **Combined insights**: Average probability used for final loan recommendation  
- **Recommendations**:  
  - ✅ Approve Loan (Very Low/Low Risk)  
  - ⚠️ Review Carefully (Moderate Risk)  
  - ❌ Reject Loan (High Risk)  

---

## 📚 Model Details  

### 🔄 Logistic Regression  
- **Preprocessing**: Scaled numerical features  
- **Output**: Probability of default (0–1)  
- **Interpretation**: Higher probability → higher risk  

### 🌳 CatBoost  
- **Features**: Categorical and numerical features automatically handled  
- **Training**: Pretrained on historical Home Credit dataset  
- **Output**: Probability of default (0–1)  
- **Interpretation**: Tree-based model captures non-linear dependencies  

---

## 📊 Evaluation Metrics  
- **Probability-based decisions**: Threshold-based risk badges  
- **Average probability**: Aggregated across selected models  
- **Visual assessment**: Gauge and card-based representation  

---

## 🌐 Live App  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://home-credit-default-risk-prediction-qk7hunfbdosspygvyqrybb.streamlit.app/)

---

## 📚 Useful Links  
- [Home Credit Default Risk Dataset](https://www.kaggle.com/c/home-credit-default-risk/data)  
- [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
- [CatBoost Documentation](https://catboost.ai/docs/)  
- [Streamlit Documentation](https://docs.streamlit.io/)  

---

> 🔖 Submitted as part of the **DevelopersHub Internship Program**
