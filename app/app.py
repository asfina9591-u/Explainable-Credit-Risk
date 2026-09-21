import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Explainable Credit Risk",
    page_icon="💳",
    layout="wide"
)


# -----------------------------
# LOAD MODEL
# -----------------------------

model = joblib.load("credit_risk_model.pkl")
feature_names = joblib.load("feature_names.pkl")


# -----------------------------
# TITLE
# -----------------------------

st.title("💳 Explainable Credit Risk Decision System")

st.write(
    "AI-powered credit risk assessment using XGBoost "
    "with explainable risk factors."
)

st.divider()


# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.header("📌 About the System")

    st.write(
        "This system estimates the probability of serious "
        "credit delinquency within two years."
    )

    st.write("**Model:** XGBoost")

    st.write("**Explainability:** SHAP / feature contribution")

    st.write("**Dataset:** Give Me Some Credit")


# -----------------------------
# APPLICANT INFORMATION
# -----------------------------

st.subheader("👤 Applicant Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=45
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=5000.0
    )

    debt_ratio = st.number_input(
        "Debt Ratio",
        min_value=0.0,
        value=0.30
    )


with col2:

    utilization = st.number_input(
        "Revolving Credit Utilization",
        min_value=0.0,
        value=0.20
    )

    late_30_59 = st.number_input(
        "30-59 Days Past Due",
        min_value=0,
        value=0
    )

    late_60_89 = st.number_input(
        "60-89 Days Past Due",
        min_value=0,
        value=0
    )


with col3:

    late_90 = st.number_input(
        "90+ Days Late",
        min_value=0,
        value=0
    )

    open_credit = st.number_input(
        "Open Credit Lines / Loans",
        min_value=0,
        value=5
    )

    real_estate = st.number_input(
        "Real Estate Loans / Lines",
        min_value=0,
        value=1
    )


dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    value=0
)


st.divider()


# -----------------------------
# PREDICTION BUTTON
# -----------------------------

if st.button(
    "🔍 Assess Credit Risk",
    use_container_width=True
):

    # Create applicant dataframe

    input_data = pd.DataFrame(
        [[
            utilization,
            age,
            late_30_59,
            debt_ratio,
            monthly_income,
            open_credit,
            late_90,
            real_estate,
            late_60_89,
            dependents
        ]],
        columns=feature_names
    )


    # -----------------------------
    # MODEL PREDICTION
    # -----------------------------

    probability = model.predict_proba(
        input_data
    )[0][1]

    risk_percentage = probability * 100


    # -----------------------------
    # RISK CATEGORY
    # -----------------------------

    if probability >= 0.50:

        risk_level = "HIGH RISK"

    else:

        risk_level = "LOW RISK"


    # -----------------------------
    # RESULT
    # -----------------------------

    st.subheader("📊 Risk Assessment")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        if probability >= 0.50:

            st.error(
                f"⚠️ {risk_level}"
            )

        else:

            st.success(
                f"✅ {risk_level}"
            )


    with result_col2:

        st.metric(
            "Risk Probability",
            f"{risk_percentage:.2f}%"
        )


    st.progress(
        float(min(probability, 1.0))
    )


    st.divider()


    # -----------------------------
    # EXPLANATION
    # -----------------------------

    st.subheader("🔎 Why did the model make this prediction?")


    booster = model.get_booster()

    contribution = booster.predict(
        xgb.DMatrix(input_data),
        pred_contribs=True
    )[0]


    shap_values = contribution[:-1]


    explanation = pd.DataFrame({

        "Feature": feature_names,

        "Value": input_data.iloc[0].values,

        "Impact": shap_values

    })


    explanation["Absolute Impact"] = (
        explanation["Impact"].abs()
    )


    explanation = explanation.sort_values(
        "Absolute Impact",
        ascending=False
    )


    # -----------------------------
    # TOP RISK FACTORS
    # -----------------------------

    st.write("### Top Factors")

    top_features = explanation.head(5)


    for _, row in top_features.iterrows():

        feature = row["Feature"]
        impact = row["Impact"]


        if impact > 0:

            st.error(
                f"🔴 **{feature}** → "
                f"increased predicted risk"
            )

        else:

            st.success(
                f"🟢 **{feature}** → "
                f"reduced predicted risk"
            )


    # -----------------------------
    # DETAIL TABLE
    # -----------------------------

    st.subheader("📋 Detailed Feature Contributions")


    display_df = explanation[
        ["Feature", "Value", "Impact"]
    ].copy()


    display_df["Impact"] = display_df[
        "Impact"
    ].round(4)


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------
    # BUSINESS ACTION
    # -----------------------------

    st.subheader("💡 Decision Support")


    if probability >= 0.50:

        st.warning(
            "The applicant shows elevated predicted "
            "delinquency risk. A financial institution "
            "could consider additional verification, "
            "risk review, or revised lending conditions."
        )

    else:

        st.info(
            "The applicant shows lower predicted "
            "delinquency risk according to the model. "
            "Standard credit review procedures can "
            "still be applied."
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