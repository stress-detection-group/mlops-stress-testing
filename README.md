#  Income Prediction for Loan Approvals using MLOps

This project implements an end-to-end **MLOps pipeline** for predicting whether an individual's income exceeds 50K using the Adult Census dataset.  
The system is designed to support **loan approval decisions** while ensuring fairness, robustness, and continuous monitoring.

---

##  Project Overview

Loan approval systems rely heavily on income prediction. Traditional approaches often focus only on accuracy and ignore important factors such as:

- Fairness (bias across demographic groups)
- Robustness (stability under small input changes)
- Continuous monitoring after deployment

This project addresses these gaps by building a complete **MLOps lifecycle pipeline**.

---

##  Dataset

- **Name:** Adult Census Income Dataset
- **Type:** Tabular data
- **Target Variable:**  
  - `<=50K`  
  - `>50K`
- **Features include:**
  - Age
  - Workclass
  - Education
  - Occupation
  - Hours per week

---

## MLOps Pipeline

The project is divided into **three phases**:

---

###  Phase 1: Modelling

- Trained multiple models:
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Naive Bayes  
  - (Optional) CNN for experimentation  

- Used **MLflow** for:
  - Experiment tracking  
  - Logging metrics (accuracy, recall)  
  - Model versioning  

---

###  Phase 2: Pre-Deployment Validation

Before deployment, the model is evaluated on:

####  Bias (Fairness)
- Metric: **Demographic Parity**
- Ensures fairness across demographic groups

####  Robustness
- Tests model sensitivity to small changes in input features  
- Predictions should remain stable

####  Prediction Quality
- Focus on **Recall (False Negatives)**  
- Important to correctly identify high-income individuals

---

###  CI/CD Pipeline

- Implemented using:
  - GitHub Actions / Jenkins  

- Workflow:
  1. Train model  
  2. Run validation tests  
  3. If all tests pass → Deploy model  

---

###  Phase 3: Post-Deployment

####  Deployment
- FastAPI for serving model  
- Docker for containerization  

####  Real-Time Simulation
- Synthetic data generation using:
  - Faker  
  - Scikit-learn  

####  Monitoring
- Log predictions and performance  
- Track model behavior over time  

---

##  Drift Detection

###  Data Drift
- Change in input data distribution over time  

###  Concept Drift
- Change in relationship between input and output  

If drift is detected, the model can be retrained.

---

##  Evaluation Metrics

- Accuracy  
- Recall (important metric)  
- Bias Score  
- Robustness Score  

---

##  Tech Stack

- **Programming Language:** Python  
- **Libraries:**  
  - Scikit-learn  
  - Pandas  
  - NumPy  
- **MLOps Tools:**  
  - MLflow  
  - DVC (optional for data versioning)  
- **Deployment:**  
  - FastAPI  
  - Docker  
- **CI/CD:**  
  - GitHub Actions / Jenkins  

