# 💳 Explainable Credit Risk Decision System

## 📌 Project Overview

The Explainable Credit Risk Decision System is an AI-powered data analytics application that estimates the probability of serious credit delinquency within two years.

The system uses machine learning to analyze applicant financial information and provides an explainable prediction using feature contributions.

The project combines:

- Data Analytics
- Machine Learning
- Explainable AI
- Interactive Data Visualization
- Streamlit

---

## 🎯 Objective

The main objective is to build a credit-risk prediction system that does not only provide a prediction, but also explains which applicant characteristics contributed to the predicted risk.

---

## 📊 Dataset

The project uses the **Give Me Some Credit** dataset.

The dataset contains information related to:

- Age
- Monthly Income
- Debt Ratio
- Revolving Credit Utilization
- Number of Open Credit Lines and Loans
- Past-Due Payment History
- Real Estate Loans
- Number of Dependents

### Target Variable

`SeriousDlqin2yrs`

This represents whether an individual experienced serious delinquency within two years.

---

## 🔎 Data Analysis

The analysis included:

- Dataset inspection
- Missing-value analysis
- Duplicate detection
- Target distribution analysis
- Feature statistics
- Outlier/anomaly investigation
- Feature importance analysis

The target variable is imbalanced, with the majority of observations belonging to the non-delinquent class.

---

## 🤖 Machine Learning

An **XGBoost classification model** was trained to estimate credit delinquency risk.

The workflow includes:

1. Data cleaning
2. Feature preparation
3. Train-test split
4. Model training
5. Model evaluation
6. Feature importance analysis
7. Explainable AI analysis

---

## 🔍 Explainable AI

The project uses feature contribution analysis to explain individual predictions.

The application identifies features that:

- Increase predicted risk
- Reduce predicted risk

This makes the model output easier to understand compared with a prediction-only system.

---

## 🖥️ Streamlit Application

The application allows users to enter applicant information and receive:

- Risk classification
- Risk probability
- Top contributing features
- Detailed feature contributions
- Decision-support information

---

## 📁 Project Structure

```text
Explainable-Credit-Risk/
│
├── app/
│   ├── app.py
│   ├── credit_risk_model.pkl
│   └── feature_names.pkl
│
├── data/
│   └── cs-training.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── outputs/
│   └── screenshots/
│
├── requirements.txt
├── README.md
└── Credit_Risk_Project_Report.pdf

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/asfina9591-u/Explainable-Credit-Risk.git