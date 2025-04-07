from flask import Flask, render_template, request, jsonify, send_file
import pickle
import os, sys
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from loandefault.exception.exception import LoanDefaultException

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Paths to model and preprocessor
MODEL_PATH = "final_model/model.pkl"
PREPROCESSOR_PATH = "final_model/preprocessor.pkl"

# Load model and preprocessor
model = pickle.load(open(MODEL_PATH, "rb")) if os.path.exists(MODEL_PATH) else None
preprocessor = pickle.load(open(PREPROCESSOR_PATH, "rb")) if os.path.exists(PREPROCESSOR_PATH) else None

# Dropdown options
OPTIONS = {
    "Loan_Purpose": ['select','Medical', 'Personal', 'Education', 'Home Improvement', 'Business'],
    "Residence_Status": ['select', 'Rented', 'Mortgaged', 'Owned'],
    "Home_Ownership": ['select','Own', 'Rent', 'Mortgage'],
    "Employment_Type": ['select', 'Retired', 'Unemployed', 'Salaried', 'Self-Employed']
}

@app.route('/')
def home():
    return render_template('index.html', **OPTIONS)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', **OPTIONS, predicted_salary=None)
    
    try:
        data = request.form if request.content_type == 'application/x-www-form-urlencoded' else request.json

        # Collect and clean inputs
        applicant_income = data.get("Applicant_Income", "").strip()
        loan_amount = data.get("Loan_Amount", "").strip()
        credit_score = data.get("Credit_Score", "").strip()
        previous_defaults = data.get("Previous_Defaults", "").strip()
        interest_rate = data.get("Interest_Rate", "").strip()
        number_of_dependents = data.get("Number_of_Dependents", "").strip()
        loan_purpose = data.get("Loan_Purpose", "").strip()
        residence_status = data.get("Residence_Status", "").strip()
        home_ownership = data.get("Home_Ownership", "").strip()
        employment_type = data.get("Employment_Type", "").strip()

        if not all([applicant_income, loan_amount, credit_score, previous_defaults, interest_rate, number_of_dependents,
                    loan_purpose, residence_status, home_ownership, employment_type]):
            return render_template('predict.html', **OPTIONS, predicted_salary="All fields are required.")

        try:
            applicant_income = float(applicant_income)
            loan_amount = float(loan_amount)
            credit_score = float(credit_score)
            previous_defaults = float(previous_defaults)
            interest_rate = float(interest_rate)
            number_of_dependents = float(number_of_dependents)
        except ValueError:
            return render_template('predict.html', **OPTIONS, predicted_salary="Invalid numerical input.")

        # Encode categorical features using index (can be improved later)
        loan_purpose_idx = OPTIONS["Loan_Purpose"].index(loan_purpose)
        residence_status_idx = OPTIONS["Residence_Status"].index(residence_status)
        home_ownership_idx = OPTIONS["Home_Ownership"].index(home_ownership)
        employment_type_idx = OPTIONS["Employment_Type"].index(employment_type)

        features_df = pd.DataFrame([[applicant_income, loan_amount, credit_score, previous_defaults, interest_rate,
                                     number_of_dependents, loan_purpose_idx, residence_status_idx, home_ownership_idx,
                                     employment_type_idx]],
                                   columns=["Applicant_Income", "Loan_Amount", "Credit_Score", "Previous_Defaults",
                                            "Interest_Rate", "Number_of_Dependents", "Loan_Purpose",
                                            "Residence_Status", "Home_Ownership", "Employment_Type"])

        if preprocessor:
            features = preprocessor.transform(features_df)
        else:
            return render_template('predict.html', **OPTIONS, predicted_salary="Preprocessor not available.")

        if model:
            prediction = model.predict(features)[0]
            confidence = model.predict_proba(features)[0][prediction] * 100
            result = f"The applicant is {'likely to DEFAULT' if prediction == 1 else 'NOT likely to default'} (Confidence: {confidence:.2f}%)."
            return render_template('predict.html', **OPTIONS, predicted_salary=result)
        else:
            return render_template('predict.html', **OPTIONS, predicted_salary="Model not available.")

    except Exception as e:
        return render_template('predict.html', **OPTIONS, predicted_salary=f"Error: {str(e)}")

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    if request.method == 'GET':
        return render_template('batch_predict.html')

    # Look for any uploaded .csv file, regardless of field name
    file = None
    for f in request.files.values():
        if f and f.filename.endswith('.csv'):
            file = f
            break

    if not file:
        return "No CSV file uploaded", 400

    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)

        # Drop unused columns
        df.drop(columns=['Marital_Status', 'Education_Level', 'Loan_Application_Type', 'State',
                         'Debt_to_Income_Ratio', 'Employment_History', 'Loan_Term', 'Missed_Payments',
                         'Credit_History_Length', 'Bankruptcies', 'Co-Applicant_Income', 'Annual_Expenses',
                         'Monthly_Installment', 'Self_Employed'],
                errors='ignore', inplace=True)

        if not preprocessor:
            return "Preprocessor not available", 500

        features = preprocessor.transform(df)

        if not model:
            return "Model not available", 500

        predictions = model.predict(features)
        confidence_scores = model.predict_proba(features).max(axis=1) * 100

        df['Default/No Default'] = np.where(predictions == 1, "Default", "No Default")
        df['Confidence (%)'] = confidence_scores.round(2)

        output_path = os.path.join(UPLOAD_FOLDER, "predictions.csv")
        df.to_csv(output_path, index=False)
        return send_file(output_path, as_attachment=True)

    except Exception as e:
        raise LoanDefaultException(e, sys)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
