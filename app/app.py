import pandas as pd
import streamlit as st
import xgboost as xgb
import joblib
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Explainable Credit Risk", page_icon="💳", layout="wide"
)


# -----------------------------
# LOAD MODEL & DATASET
# -----------------------------

model = joblib.load("app/credit_risk_model.pkl")
feature_names = joblib.load("app/feature_names.pkl")

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("data/cs-training.csv")

# -----------------------------
# MODEL EVALUATION DATA
# -----------------------------

evaluation_df = df.drop_duplicates().copy()
X_eval = evaluation_df[feature_names]
y_eval = evaluation_df["SeriousDlqin2yrs"]

X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
    X_eval, y_eval, test_size=0.20, random_state=42, stratify=y_eval
)

y_probability = model.predict_proba(X_test_eval)[:, 1]
y_prediction = (y_probability >= 0.50).astype(int)

roc_auc = roc_auc_score(y_test_eval, y_probability)
pr_auc = average_precision_score(y_test_eval, y_probability)
recall = recall_score(y_test_eval, y_prediction)
cm = confusion_matrix(y_test_eval, y_prediction)


# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------

st.sidebar.title("💳 Credit Risk System")

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Risk Drivers",
        "Decision Simulator",
        "Recommendations",
    ],
)

st.sidebar.divider()

st.sidebar.write("**Model:** XGBoost")
st.sidebar.write("**Explainability:** SHAP")
st.sidebar.write("**Dataset:** Give Me Some Credit")


# ============================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    st.title("📊 Executive Overview")

    st.write("Key analytics from the Give Me Some Credit dataset.")

    # -----------------------------
    # KPI CALCULATIONS
    # -----------------------------

    default_rate = df["SeriousDlqin2yrs"].mean() * 100

    average_age = df["age"].mean()

    average_income = df["MonthlyIncome"].mean()

    high_risk_percentage = (df["SeriousDlqin2yrs"] == 1).mean() * 100

    # -----------------------------
    # KPI CARDS
    # -----------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Default Rate", f"{default_rate:.2f}%")

    col2.metric("Average Age", f"{average_age:.1f}")

    col3.metric("Average Income", f"${average_income:,.0f}")

    col4.metric("High-Risk %", f"{high_risk_percentage:.2f}%")

    st.divider()

    # -----------------------------
    # AGE BAND ANALYSIS
    # -----------------------------

    st.subheader("📈 Default Rate by Age Band")

    age_bins = [0, 30, 40, 50, 60, 70, 120]

    age_labels = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]

    df["Age Band"] = pd.cut(
        df["age"], bins=age_bins, labels=age_labels, right=False
    )

    age_default = (
        df.groupby("Age Band", observed=True)["SeriousDlqin2yrs"]
        .mean()
        .mul(100)
        .reset_index()
    )

    age_default.columns = ["Age Band", "Default Rate"]

    st.bar_chart(age_default.set_index("Age Band"))

    st.caption(
        "Default rate represents the percentage of applicants "
        "in each age band who experienced serious delinquency "
        "within two years."
    )

    st.divider()

    # -----------------------------
    # MODEL PERFORMANCE
    # -----------------------------

    st.subheader("🤖 Model Performance")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("ROC-AUC", f"{roc_auc:.3f}")

    metric2.metric("PR-AUC", f"{pr_auc:.3f}")

    metric3.metric("Recall", f"{recall:.3f}")

    st.caption(
        "ROC-AUC measures ranking ability across classification "
        "thresholds. PR-AUC focuses on performance for the "
        "positive class. Recall measures how many actual "
        "delinquent applicants were identified."
    )

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------

    st.subheader("📋 Confusion Matrix")

    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Delinquency", "Actual: Delinquency"],
        columns=["Predicted: No Delinquency", "Predicted: Delinquency"],
    )

    st.dataframe(cm_df, use_container_width=True)


# ============================================================
# PAGE 2 — RISK DRIVERS
# ============================================================

elif page == "Risk Drivers":

    st.title("🔎 Risk Drivers")

    st.write(
        "Analysis of the factors that influence credit-risk predictions."
    )

    # -----------------------------
    # FRIENDLY FEATURE NAMES
    # -----------------------------

    friendly_names = {
        "RevolvingUtilizationOfUnsecuredLines": "Credit Utilization",
        "age": "Age",
        "NumberOfTime30-59DaysPastDueNotWorse": "30-59 Days Late",
        "DebtRatio": "Debt Ratio",
        "MonthlyIncome": "Monthly Income",
        "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines / Loans",
        "NumberOfTimes90DaysLate": "90+ Days Late",
        "NumberRealEstateLoansOrLines": "Real Estate Loans",
        "NumberOfTime60-89DaysPastDueNotWorse": "60-89 Days Late",
        "NumberOfDependents": "Dependents",
    }

    # -----------------------------
    # GLOBAL FEATURE IMPORTANCE
    # -----------------------------

    st.subheader("📊 Global Feature Importance")

    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importance}
    )

    importance_df["Feature"] = importance_df["Feature"].map(friendly_names)

    importance_df = importance_df.sort_values("Importance", ascending=True)

    st.bar_chart(importance_df.set_index("Feature"))

    st.caption(
        "Higher importance indicates that the feature was used "
        "more heavily by the XGBoost model when making predictions."
    )

    st.divider()

    # -----------------------------
    # UTILIZATION BAND ANALYSIS
    # -----------------------------

    st.subheader("📈 Default Rate by Credit Utilization")

    utilization_bins = [-0.01, 0.10, 0.30, 0.50, 1.00, float("inf")]

    utilization_labels = ["0-10%", "10-30%", "30-50%", "50-100%", "100%+"]

    df["Utilization Band"] = pd.cut(
        df["RevolvingUtilizationOfUnsecuredLines"],
        bins=utilization_bins,
        labels=utilization_labels,
    )

    utilization_default = (
        df.groupby("Utilization Band", observed=True)["SeriousDlqin2yrs"]
        .mean()
        .mul(100)
        .reset_index()
    )

    utilization_default.columns = ["Utilization Band", "Default Rate"]

    st.bar_chart(utilization_default.set_index("Utilization Band"))

    st.caption(
        "Default rate represents the percentage of applicants "
        "within each credit-utilization band who experienced "
        "serious delinquency within two years."
    )

    st.divider()

    # -----------------------------
    # SHAP EXPLANATION
    # -----------------------------

    st.subheader("🧠 Understanding Feature Impact")

    st.write(
        "The Decision Simulator uses feature contributions "
        "to explain individual predictions."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            "🟢 Negative contribution → lowers the model's "
            "predicted risk."
        )

    with col2:

        st.error(
            "🔴 Positive contribution → increases the model's "
            "predicted risk."
        )


# ============================================================
# PAGE 3 — DECISION SIMULATOR
# ============================================================

elif page == "Decision Simulator":

    st.title("💳 Explainable Credit Risk Decision System")

    st.write(
        "AI-powered credit risk assessment using XGBoost "
        "with explainable risk factors."
    )

    st.divider()

    # -----------------------------
    # DECISION THRESHOLD
    # -----------------------------

    st.subheader("⚙️ Decision Threshold")

    st.write(
        "Adjust the relative business cost of missing a "
        "potentially delinquent applicant versus incorrectly "
        "flagging a non-delinquent applicant."
    )

    cost_fn = st.slider(
        "Cost of Missing a Delinquent Applicant",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    cost_fp = st.slider(
        "Cost of Incorrectly Flagging a Non-Delinquent Applicant",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
    )

    threshold = cost_fp / (cost_fn + cost_fp)

    st.info(
        f"Calculated decision threshold: "
        f"**{threshold:.2f} ({threshold * 100:.1f}%)**"
    )

    st.divider()

    # -----------------------------
    # APPLICANT INFORMATION
    # -----------------------------

    st.subheader("👤 Applicant Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age", min_value=18, max_value=100, value=45
        )

        monthly_income = st.number_input(
            "Monthly Income", min_value=0.0, value=5000.0
        )

        debt_ratio = st.number_input(
            "Debt Ratio", min_value=0.0, value=0.30
        )

    with col2:

        utilization = st.number_input(
            "Revolving Credit Utilization", min_value=0.0, value=0.20
        )

        late_30_59 = st.number_input(
            "30-59 Days Past Due", min_value=0, value=0
        )

        late_60_89 = st.number_input(
            "60-89 Days Past Due", min_value=0, value=0
        )

    with col3:

        late_90 = st.number_input("90+ Days Late", min_value=0, value=0)

        open_credit = st.number_input(
            "Open Credit Lines / Loans", min_value=0, value=5
        )

        real_estate = st.number_input(
            "Real Estate Loans / Lines", min_value=0, value=1
        )

    dependents = st.number_input(
        "Number of Dependents", min_value=0, value=0
    )

    st.divider()

    # -----------------------------
    # PREDICTION
    # -----------------------------

    if st.button("🔍 Assess Credit Risk", use_container_width=True):

        input_data = pd.DataFrame(
            [
                [
                    utilization,
                    age,
                    late_30_59,
                    debt_ratio,
                    monthly_income,
                    open_credit,
                    late_90,
                    real_estate,
                    late_60_89,
                    dependents,
                ]
            ],
            columns=feature_names,
        )

        probability = model.predict_proba(input_data)[0][1]

        risk_percentage = probability * 100

        # -----------------------------
        # DECISION BANDS
        # -----------------------------

        if probability < threshold * 0.6:
            decision = "APPROVE"
        elif probability < threshold:
            decision = "MANUAL REVIEW"
        else:
            decision = "REJECT"

        # -----------------------------
        # RESULT
        # -----------------------------

        st.subheader("📊 Risk Assessment")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if decision == "APPROVE":
                st.success("🟢 APPROVE")
            elif decision == "MANUAL REVIEW":
                st.warning("🟡 MANUAL REVIEW")
            else:
                st.error("🔴 REJECT")

        with result_col2:

            st.metric(
                "Risk Probability", f"{risk_percentage:.2f}%"
            )

        st.progress(float(min(probability, 1.0)))

        st.divider()

        # -----------------------------
        # SHAP EXPLANATION
        # -----------------------------

        st.subheader("🔎 Why did the model make this prediction?")

        booster = model.get_booster()

        contribution = booster.predict(
            xgb.DMatrix(input_data), pred_contribs=True
        )[0]

        shap_values = contribution[:-1]

        explanation = pd.DataFrame(
            {
                "Feature": feature_names,
                "Value": input_data.iloc[0].values,
                "Impact": shap_values,
            }
        )

        explanation["Absolute Impact"] = explanation["Impact"].abs()

        explanation = explanation.sort_values(
            "Absolute Impact", ascending=False
        )

        # -----------------------------
        # TOP FACTORS
        # -----------------------------

        st.write("### Top Factors")

        top_features = explanation.head(5)

        for _, row in top_features.iterrows():

            feature = row["Feature"]
            impact = row["Impact"]

            if impact > 0:

                st.error(
                    f"🔴 **{feature}** → " f"increased predicted risk"
                )

            else:

                st.success(
                    f"🟢 **{feature}** → " f"reduced predicted risk"
                )

        # -----------------------------
        # DETAIL TABLE
        # -----------------------------

        st.subheader("📋 Detailed Feature Contributions")

        display_df = explanation[["Feature", "Value", "Impact"]].copy()

        display_df["Impact"] = display_df["Impact"].round(4)

        st.dataframe(
            display_df, use_container_width=True, hide_index=True
        )

        # -----------------------------
        # DECISION SUPPORT
        # -----------------------------

        st.subheader("💡 Decision Support")

        if decision == "APPROVE":

            st.success(
                "The applicant falls below the configured "
                "risk threshold. The model indicates relatively "
                "lower predicted delinquency risk."
            )

        elif decision == "MANUAL REVIEW":

            st.warning(
                "The applicant falls near the configured "
                "decision boundary. Additional financial "
                "verification or manual review may be appropriate."
            )

        else:

            st.error(
                "The applicant exceeds the configured risk "
                "threshold. The model indicates elevated "
                "predicted delinquency risk."
            )


# ============================================================
# PAGE 4 — RECOMMENDATIONS
# ============================================================

elif page == "Recommendations":

    st.title("💡 Recommendations")

    st.write(
        "Data-driven actions based on the observed risk "
        "patterns and model insights."
    )

    st.divider()

    # -----------------------------
    # RECOMMENDATION TABLE
    # -----------------------------

    recommendations = pd.DataFrame(
        {
            "Area": [
                "Credit Utilization",
                "Payment History",
                "Debt Burden",
                "Risk Assessment",
            ],
            "Risk": [
                "Higher utilization may indicate greater credit exposure.",
                "Repeated late payments can indicate higher delinquency risk.",
                "Higher debt burden may reduce repayment capacity.",
                "A single probability threshold may not capture every borderline applicant.",
            ],
            "Opportunity": [
                "Monitor applicants with high revolving credit utilization.",
                "Use payment history as an important risk-review signal.",
                "Combine debt ratio with income and other financial indicators.",
                "Use explainable predictions to support manual review.",
            ],
            "Action": [
                "Flag high-utilization applicants for additional review.",
                "Review recent and repeated delinquency indicators.",
                "Consider income and debt indicators together.",
                "Use the configurable threshold and SHAP explanations as decision-support tools.",
            ],
        }
    )

    st.dataframe(
        recommendations, use_container_width=True, hide_index=True
    )

    st.divider()

    # -----------------------------
    # KEY INSIGHTS
    # -----------------------------

    st.subheader("📌 Key Insights")

    st.markdown(
        """
        **1. Payment behaviour matters**

        Past-due payment variables are important signals
        for identifying potential delinquency risk.

        **2. Credit utilization should be monitored**

        Applicants with higher revolving credit utilization
        can be investigated more closely during risk assessment.

        **3. Model explanations improve transparency**

        SHAP-based feature contributions help explain why
        an individual applicant received a particular
        prediction.

        **4. Borderline cases require additional review**

        The system provides a configurable threshold so
        that borderline cases can be directed toward
        manual review rather than relying only on a
        binary prediction.
        """
    )

    st.divider()

    st.info(
        "These recommendations are analytical insights "
        "from the dataset and model. They are not "
        "financial or lending advice."
    )


# -----------------------------
# FOOTER
# -----------------------------

st.divider()

st.caption(
    "⚠️ This system is an analytical decision-support "
    "tool and should not be used as the sole basis for "
    "real-world lending decisions."
)