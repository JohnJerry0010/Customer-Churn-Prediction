from flask import Flask, request, render_template, Blueprint
import joblib
import pandas as pd
import os

# Initialize the Flask app
app = Flask(__name__)

# Define the paths to the model, scaler, label encoders, and top features
model_path = r"C:\Users\hp\OneDrive\Documents\attrition\model\best_model_smote_XGBoost.pkl"
scaler_path = r"C:\Users\hp\OneDrive\Documents\attrition\model\scaler.pkl"
le_path = r"C:\Users\hp\OneDrive\Documents\attrition\model\label_encoders.pkl"  # This should be a dict of encoders
top_features_path = r"C:\Users\hp\OneDrive\Documents\attrition\model\top_features.pkl"

# Check if the files exist
for path in [model_path, scaler_path, le_path, top_features_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

# Load all required artifacts
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
le_dict = joblib.load(le_path)  # Make sure this is a dict: {col: LabelEncoder()}
top_features = joblib.load(top_features_path)

# Define blueprint
main_routes = Blueprint('main_routes', __name__)

# Route to render the home page
@main_routes.route('/')
def index():
    return render_template('index.html')

# Route to render the form
@main_routes.route('/submit', methods=['GET'])
def show_form():
    return render_template('forms.html')
"""
# Route to handle form submission
@main_routes.route('/submit', methods=['POST'])
def handle_form_submission():
    try:
        # Define required fields
        required_fields = ['OverTime', 'MonthlyIncome', 'YearsWithCurrManager', 'YearsSinceLastPromotion', 
                           'SalarySlab', 'EnvironmentSatisfaction', 'StockOptionLevel', 'YearsInCurrentRole', 
                           'Department', 'JobSatisfaction', 'YearsAtCompany', 'TotalWorkingYears', 
                           'JobInvolvement', 'NumCompaniesWorked', 'Age']

        # Check if all required fields are present
        if not all(field in request.form for field in required_fields):
            return "Error: Missing required fields in the form."

        # Extract and convert form data
        form_data = {
            'Age': int(request.form['Age']),
            'Department': request.form['Department'],
            'OverTime': request.form['OverTime'],
            'YearsWithCurrManager': int(request.form['YearsWithCurrManager']),
            'YearsSinceLastPromotion': int(request.form['YearsSinceLastPromotion']),
            'YearsInCurrentRole': int(request.form['YearsInCurrentRole']),
            'YearsAtCompany': int(request.form['YearsAtCompany']),
            'TotalWorkingYears': int(request.form['TotalWorkingYears']),
            'NumCompaniesWorked': int(request.form['NumCompaniesWorked']),
            'MonthlyIncome': float(request.form['MonthlyIncome']),
            'SalarySlab': request.form['SalarySlab'],
            'StockOptionLevel': int(request.form['StockOptionLevel']),
            'EnvironmentSatisfaction': int(request.form['EnvironmentSatisfaction']),
            'JobSatisfaction': int(request.form['JobSatisfaction']),
            'JobInvolvement': int(request.form['JobInvolvement']),
        }

        # Convert form data into DataFrame
        df = pd.DataFrame([form_data])

        # Encode categorical columns using the dict of label encoders
        for col in df.select_dtypes(include='object').columns:
            if col in le_dict:
                df[col] = le_dict[col].transform(df[col])
            else:
                return f"Encoder for column '{col}' not found."

        # Ensure columns are in correct order according to top features
        df = df[top_features]

        # Scale the input features
        scaled_input = scaler.transform(df)

        # Make prediction using the model
        prediction = model.predict(scaled_input)[0]
        result = "Yes, the employee may churn." if prediction == 1 else "No, the employee is likely to stay."

        # Return the result page with the prediction
        return render_template('result.html', result=result)

    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

# Register the blueprint
app.register_blueprint(main_routes)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
    
    """
    
"""
@main_routes.route('/submit', methods=['POST'])
def handle_form_submission():
    try:
        # Define required fields
        required_fields = ['OverTime', 'MonthlyIncome', 'YearsWithCurrManager', 'YearsSinceLastPromotion',
                           'SalarySlab', 'EnvironmentSatisfaction', 'StockOptionLevel', 'YearsInCurrentRole',
                           'Department', 'JobSatisfaction', 'YearsAtCompany', 'TotalWorkingYears',
                           'JobInvolvement', 'NumCompaniesWorked', 'Age']

        # Check if all required fields are present
        if not all(field in request.form for field in required_fields):
            return "Error: Missing required fields in the form."

        # Extract form data
        form_data = {
            'Age': int(request.form['Age']),
            'Department': request.form['Department'],
            'OverTime': request.form['OverTime'],
            'YearsWithCurrManager': int(request.form['YearsWithCurrManager']),
            'YearsSinceLastPromotion': int(request.form['YearsSinceLastPromotion']),
            'YearsInCurrentRole': int(request.form['YearsInCurrentRole']),
            'YearsAtCompany': int(request.form['YearsAtCompany']),
            'TotalWorkingYears': int(request.form['TotalWorkingYears']),
            'NumCompaniesWorked': int(request.form['NumCompaniesWorked']),
            'MonthlyIncome': float(request.form['MonthlyIncome']),
            'SalarySlab': request.form['SalarySlab'],
            'StockOptionLevel': int(request.form['StockOptionLevel']),
            'EnvironmentSatisfaction': int(request.form['EnvironmentSatisfaction']),
            'JobSatisfaction': int(request.form['JobSatisfaction']),
            'JobInvolvement': int(request.form['JobInvolvement']),
        }

        # Convert to DataFrame
        df = pd.DataFrame([form_data])

        # Encode categorical columns
        for col in df.select_dtypes(include='object').columns:
            if col in le_dict:
                df[col] = le_dict[col].transform(df[col])
            else:
                return f"Encoder for column '{col}' not found."

        # Select top features
        df = df[top_features]

        # Scale
        scaled_input = scaler.transform(df)

        # Prediction and confidence
        prediction = model.predict(scaled_input)[0]
        confidence = model.predict_proba(scaled_input)[0][prediction] * 100  # get probability for the predicted class

        result = "Yes, the employee may churn." if prediction == 1 else "No, the employee is likely to stay."

        # Generate insights
        insights = []
        if form_data['OverTime'] == 'Yes':
            insights.append("The employee is doing overtime which may lead to burnout.")
        if form_data['MonthlyIncome'] < 4000:
            insights.append("Low monthly income might be contributing to attrition risk.")
        if form_data['YearsSinceLastPromotion'] > 3:
            insights.append("The employee hasn't been promoted for a long time.")
        if form_data['JobSatisfaction'] <= 2:
            insights.append("The employee reports low job satisfaction.")
        if form_data['EnvironmentSatisfaction'] <= 2:
            insights.append("Poor workplace environment satisfaction might be a factor.")
        if form_data['NumCompaniesWorked'] > 4:
            insights.append("The employee has changed multiple jobs, indicating less stability.")

        return render_template(
            'result.html',
            result=result,
            confidence=round(confidence, 2),
            insights=insights
        )

    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

"""

import shap
import numpy as np

# Create a global SHAP explainer (ideally done only once during app setup if using tree-based models)
explainer = shap.Explainer(model)

@main_routes.route('/submit', methods=['POST'])
def handle_form_submission():
    try:
        # Define required fields
        required_fields = ['OverTime', 'MonthlyIncome', 'YearsWithCurrManager', 'YearsSinceLastPromotion',
                           'SalarySlab', 'EnvironmentSatisfaction', 'StockOptionLevel', 'YearsInCurrentRole',
                           'Department', 'JobSatisfaction', 'YearsAtCompany', 'TotalWorkingYears',
                           'JobInvolvement', 'NumCompaniesWorked', 'Age']

        # Check if all required fields are present
        if not all(field in request.form for field in required_fields):
            return "Error: Missing required fields in the form."

        # Extract and convert form data
        form_data = {
            'Age': int(request.form['Age']),
            'Department': request.form['Department'],
            'OverTime': request.form['OverTime'],
            'YearsWithCurrManager': int(request.form['YearsWithCurrManager']),
            'YearsSinceLastPromotion': int(request.form['YearsSinceLastPromotion']),
            'YearsInCurrentRole': int(request.form['YearsInCurrentRole']),
            'YearsAtCompany': int(request.form['YearsAtCompany']),
            'TotalWorkingYears': int(request.form['TotalWorkingYears']),
            'NumCompaniesWorked': int(request.form['NumCompaniesWorked']),
            'MonthlyIncome': float(request.form['MonthlyIncome']),
            'SalarySlab': request.form['SalarySlab'],
            'StockOptionLevel': int(request.form['StockOptionLevel']),
            'EnvironmentSatisfaction': int(request.form['EnvironmentSatisfaction']),
            'JobSatisfaction': int(request.form['JobSatisfaction']),
            'JobInvolvement': int(request.form['JobInvolvement']),
        }

        # Convert to DataFrame
        df = pd.DataFrame([form_data])

        # Encode categorical columns
        for col in df.select_dtypes(include='object').columns:
            if col in le_dict:
                df[col] = le_dict[col].transform(df[col])
            else:
                return f"Encoder for column '{col}' not found."

        # Select top features
        df = df[top_features]

        # Scale input
        scaled_input = scaler.transform(df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        confidence = model.predict_proba(scaled_input)[0][prediction] * 100

        # SHAP: Get feature contributions
        shap_values = explainer(scaled_input)
        shap_vals = shap_values.values[0]
        top_indices = np.argsort(np.abs(shap_vals))[::-1][:3]  # top 3 influential features

        # Generate SHAP-based insights
        insights = []
        for i in top_indices:
            feature = df.columns[i]
            contribution = shap_vals[i]
            direction = "increases" if contribution > 0 else "decreases"
            insights.append(f"Feature '{feature}' {direction} the likelihood of attrition.")

        # Final result string
        result = "Yes, the employee may churn." if prediction == 1 else "No, the employee is likely to stay."

        return render_template(
            'result.html',
            result=result,
            confidence=round(confidence, 2),
            insights=insights
        )

    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"
