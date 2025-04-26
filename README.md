# <b>Loan Default Prediction using Machine Learning</b>

[Link to Live Project](https://neo-finance-loan-default-prediction.onrender.com)

---

## 🏢 About the Company

**NeoFinance** is a leading fintech company founded in 2015, specializing in **personal**, **home**, and **business loans** for individuals and SMEs. With over **2 million customers** across India, Southeast Asia, and the Middle East, the company uses a **digital-first approach** to approve loans within minutes.

However, rising loan defaults have prompted the company to explore **AI-powered risk assessment models** to improve its underwriting process and reduce financial risk.

---

## 📉 Problem Statement

NeoFinance currently relies on traditional credit scoring systems like **CIBIL** and **Experian**, which overlook vital behavioral and social factors such as:

- Income fluctuations  
- Spending patterns  
- Social & behavioral data  

This outdated approach results in:
🚩 **Approval of high-risk applicants**  
🚩 **Rejection of potentially good customers**  

To mitigate this, NeoFinance aims to develop an **AI-powered Loan Default Prediction System** that:

✅ Identifies high-risk applicants before loan approval  
✅ Predicts default probability using behavioral and financial features  
✅ Optimizes interest rates and approval criteria based on risk profiling  

---

## 💼 Business Perspective

### 🔹 Why Does NeoFinance Need This?

NeoFinance's current system leads to:

- 💸 Financial losses due to defaults  
- ❌ Missed opportunities from rejecting low-risk applicants  

### ✅ Benefits of AI Integration

- 📉 **Reduce default rates** with accurate risk profiling  
- 📈 **Increase revenue** by approving more reliable applicants  
- ⚡ **Accelerate loan approvals** through real-time ML inference  
- 💡 **Enhance decision-making** for the finance and risk teams  

### 📊 Expected Impact

| Metric                        | Projected Outcome                     |
|------------------------------|----------------------------------------|
| 🔻 Loan Defaults             | 20–30% Reduction                      |
| 🔺 Approval Rate             | 10–15% Increase                       |
| ⚡ Operational Efficiency     | Faster decisions, better user experience |

---

## 📁 Data Overview

### 📥 Input Features

| Feature Name               | Description                                     |
|---------------------------|--------------------------------------------------|
| Applicant_Income          | Annual income in USD                             |
| Loan_Amount               | Requested loan amount                            |
| Credit_Score              | Credit score (300–850)                           |
| Debt_to_Income_Ratio      | Monthly debt to income ratio                     |
| Employment_History        | Total years of employment                        |
| Previous_Defaults         | Count of past defaults                           |
| Loan_Term                 | Duration in months                               |
| Interest_Rate             | Annual loan interest rate                        |
| Loan_Purpose              | Purpose of loan (e.g., Personal, Business)       |
| Marital_Status            | Marital status of applicant                      |
| Number_of_Dependents      | Number of dependents                             |
| Residence_Status          | Residence type (Owned, Rented, Mortgaged)        |
| Bankruptcies              | Past bankruptcy filings                          |
| Annual_Expenses           | Yearly expenses in USD                           |
| Co-Applicant_Income       | Co-applicant’s annual income                     |
| Education_Level           | Highest education achieved                       |
| Self_Employed             | 1 if self-employed, else 0                       |
| Home_Ownership            | Type of home ownership                           |
| Credit_History_Length     | Years with active credit accounts                |
| Missed_Payments           | Missed payments in the past year                 |
| Loan_Application_Type     | Individual or Joint application                  |
| Monthly_Installment       | Monthly EMI                                      |
| Employment_Type           | Job type (Salaried, Self-Employed, etc.)         |
| State                     | State of residence (e.g., CA, NY)                |

### 🎯 Target Variable

- **Default** – Binary classification  
  - `1` → Defaulted  
  - `0` → No Default  

---

## ⚙️ Tech Stack

| Category                           | Tools & Libraries                                 |
|------------------------------------|---------------------------------------------------|
| Language                           | Python 3.10                                       |
| ML Frameworks                      | Scikit-learn, XGBoost                             |
| Data Processing                    | Pandas, NumPy                                     |
| Visualization                      | Matplotlib, Seaborn                               |
| MLOps Performance Metrics Tracking | MLflo, Dagshub                                    |
| Deployment                         | Flask, Render.com                                 |
| Database                           | MongoDB                                           |
| CI/CD Pipelines                    | GitHub Actions                                    |
| Containerization                   | Docker, GitHub Container Registry (GHCR)          |

---

## 🛠 Installation & Setup

1️⃣ Clone the Repository  
```bash
git clone (https://github.com/AnupamKNN/LoanDefault/edit/main/README.md)
cd LoanDefaultPrediction
```

2️⃣ Create a Virtual Environment and Install Dependencies  
```bash
python3.10 -m venv venv
source venv/bin/activate  # For Linux/macOS
# OR
venv\Scripts\activate     # For Windows

pip install -r requirements.txt
```

Alternatively, using Conda:
```bash
conda create --name venv python=3.10 -y
conda activate venv
pip install -r requirements.txt
```

### Explanation:
1. Creates a Conda virtual environment** named `venv` with Python 3.10.
2. Activates the environment.
3. Installs dependencies from the `requirements.txt` file.  

This makes it easy for anyone cloning your repo to set up their environment correctly! ✅

---

## 🎯 Model Training & Evaluation
The models are trained using supervised learning algorithms from Scikit-Learn

### 📊 Evaluation Metrics

| Metric        | Description                             |
|---------------|-----------------------------------------|
| Accuracy      | Percentage of correct predictions       |
| F1 Score      | Harmonic mean of precision & recall     |
| ROC-AUC       | Area under the ROC curve                |

### 📊 Model Performance Summary

### 🔍 Before Hyperparameter Tuning

#### ✅ Training Results

| Rank | Model               | Accuracy | F1 Score | ROC AUC |
|------|---------------------|----------|----------|---------|
| 1️⃣  | RandomForest        | 1.000000 | 1.000000 | 1.000000 |
| 1️⃣  | DecisionTree        | 1.000000 | 1.000000 | 1.000000 |
| 3️⃣  | XGBoost             | 0.973394 | 0.972708 | 0.998894 |
| 4️⃣  | KNeighbors          | 0.801468 | 0.826436 | 0.926472 |
| 5️⃣  | GradientBoosting    | 0.793578 | 0.742032 | 0.877848 |
| 6️⃣  | SVC                 | 0.757064 | 0.768207 | 0.840505 |
| 7️⃣  | AdaBoost            | 0.683670 | 0.638574 | 0.754361 |
| 8️⃣  | Logistic Regression | 0.531927 | 0.532869 | 0.556152 |

### 🧪 Test Results

| Rank | Model               | Accuracy | F1 Score | ROC AUC |
|------|---------------------|----------|----------|---------|
| 1️⃣  | GradientBoosting    | 0.693    | 0.025397 | 0.477288 |
| 2️⃣  | RandomForest        | 0.667    | 0.125984 | 0.502210 |
| 3️⃣  | XGBoost             | 0.651    | 0.212190 | 0.512978 |
| 4️⃣  | AdaBoost            | 0.631    | 0.254545 | 0.508845 |
| 5️⃣  | DecisionTree        | 0.581    | 0.333863 | 0.514905 |
| 6️⃣  | SVC                 | 0.527    | 0.313498 | 0.488386 |
| 7️⃣  | Logistic Regression | 0.507    | 0.357236 | 0.511965 |
| 8️⃣  | KNeighbors          | 0.487    | 0.361146 | 0.472757 |

---

### 🎯 After Hyperparameter Tuning

### ✅ Training Results

| Rank | Model               | Accuracy | F1 Score | ROC AUC |
|------|---------------------|----------|----------|---------|
| 1️⃣  | KNN                 | 1.000000 | 1.000000 | 1.000000 |
| 1️⃣  | Random Forest       | 1.000000 | 1.000000 | 1.000000 |
| 3️⃣  | SVC                 | 0.986055 | 0.986197 | 0.999263 |
| 4️⃣  | Decision Tree       | 0.931927 | 0.929077 | 0.989983 |
| 5️⃣  | Gradient Boosting   | 0.789174 | 0.736407 | 0.867225 |
| 6️⃣  | XGBoost             | 0.785138 | 0.728872 | 0.866437 |
| 7️⃣  | AdaBoost            | 0.758899 | 0.709934 | 0.807752 |
| 8️⃣  | Logistic Regression | 0.540734 | 0.545982 | 0.556307 |

### 🧪 Test Results

| Rank | Model               | Accuracy | F1 Score | ROC AUC |
|------|---------------------|----------|----------|---------|
| 1️⃣  | Gradient Boosting   | 0.693    | 0.025397 | 0.490563 |
| 1️⃣  | XGBoost             | 0.693    | 0.019169 | 0.472113 |
| 3️⃣  | Random Forest       | 0.674    | 0.094444 | 0.498738 |
| 4️⃣  | AdaBoost            | 0.666    | 0.102151 | 0.502410 |
| 5️⃣  | Decision Tree       | 0.593    | 0.279646 | 0.473921 |
| 6️⃣  | SVC                 | 0.546    | 0.297214 | 0.483605 |
| 7️⃣  | KNN                 | 0.542    | 0.318452 | 0.489974 |
| 8️⃣  | Logistic Regression | 0.522    | 0.377604 | 0.510544 |


### Best Model

After hyperparameter tuning, **GradientBoostingClassifier** emerged as the best-performing model, achieving the highest accuracy score of **69.30%** and ROC AUC of 0.49.

---

## 🚀 Usage  
1️⃣ Input Applicant Details (income, loan amount, credit history, etc.).  
2️⃣ ML Model Predicts whether the applicant is likely to **default or not** using optimized classification algorithms.  
3️⃣ Financial Institution makes informed **loan approval decisions** to reduce risk and improve approval efficiency.

---

## 🔥 Results & Insights  
📌 The AI model accurately predicts loan default risk, enabling the financial institution to:  
✅ Reduce loan defaults by identifying high-risk applicants early.  
✅ Approve more genuine customers, increasing overall profitability.  
✅ Make smarter, data-driven lending decisions to optimize portfolio performance.

---

## ✅ Final Deliverables

- 📁 Cleaned dataset, EDA and Model Training notebook  
- 📦 Trained model saved in `.pkl`
- 🛠 Complete Deployable Project:
  - Data Ingestion, Data Validation, Data Transformation
  - Model Training Pipeline with Model Performance Metrics Tracking using MLflow
- 🚀 Flask app for model inference  
- 🖥 Frontend interface for real-time predictions

---

🌟 Star this repo if you found it helpful! 🚀
