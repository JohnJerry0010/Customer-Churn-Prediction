# 📉 Customer Churn Prediction

This project focuses on identifying customers who are likely to stop using a service (churn) using machine learning models. It is a research-based implementation that combines data preprocessing, feature selection, multiple classification algorithms, model evaluation, and a Flask-based deployment prototype.

---

## 🔍 Objective

To build a predictive system that helps businesses anticipate customer churn and take proactive steps to retain customers. The model identifies key behavioral and demographic features that influence churn and provides interpretable results to stakeholders.

---

## 📦 Key Components

### 1. 🧹 Data Preprocessing
- Handled missing values, outliers, and categorical encoding.
- Standardization and normalization of features for model compatibility.
- Label encoding of binary categories and one-hot encoding for multiclass features.

### 2. 📊 Exploratory Data Analysis (EDA)
- Univariate and bivariate analysis to understand customer behavior.
- Correlation matrix to identify multicollinearity.
- Visualizations to highlight patterns between churn and customer features.

### 3. 🎯 Feature Engineering & Selection
- Created new features such as tenure groups, engagement scores, etc.
- Feature importance ranking using:
  - XGBoost
- Selected top N features to improve model interpretability and reduce overfitting.

### 4. 🤖 Model Building
Multiple models were trained and compared:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost


Each model was evaluated using:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
---


---

## 🌐 Flask-Based Web Deployment

A lightweight Flask app was built to serve the model and allow real-time predictions.

### Features:
- User-friendly form to input customer attributes.
- Backend inference using the best-performing model (e.g., XGBoost or Random Forest).
- Displays churn probability and prediction (Churn / No Churn).

To run the app:

```bash
cd flask_app/
python app.py
