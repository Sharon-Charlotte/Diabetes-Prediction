
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "/content/diabetes-prediction-capstone/models/diabetes_logistic_pipeline.joblib"

model = joblib.load(MODEL_PATH)

# Model threshold selected during evaluation
THRESHOLD = 0.20


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("Diabetes Prediction Dashboard")

st.write(
    "A machine-learning dashboard developed using the "
    "CDC 2021 Behavioral Risk Factor Surveillance System (BRFSS) data."
)

st.info(
    "This tool demonstrates the trained project model. "
    "It is intended for analytical and educational purposes and "
    "not for clinical diagnosis."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Project Information")

st.sidebar.write("**Dataset:** CDC BRFSS 2021")
st.sidebar.write("**Model:** Logistic Regression")
st.sidebar.write("**Predictors:** 18")
st.sidebar.write("**Decision threshold:** 0.20")
st.sidebar.write("**ROC-AUC:** 0.829")
st.sidebar.write("**PR-AUC:** 0.423")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Model Performance",
    "🔮 Diabetes Prediction",
    "ℹ️ About the Project"
])


# ============================================================
# TAB 1 — MODEL PERFORMANCE
# ============================================================

with tab1:

    st.header("Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", "79.44%")

    with col2:
        st.metric("Precision", "35.99%")

    with col3:
        st.metric("Recall", "65.86%")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("F1 Score", "46.55%")

    with col5:
        st.metric("ROC-AUC", "82.86%")

    with col6:
        st.metric("PR-AUC", "42.27%")

    st.subheader("Confusion Matrix")

    cm = pd.DataFrame(
        [
            [59774, 13495],
            [3934, 7589]
        ],
        index=["Actual No Diabetes", "Actual Diabetes"],
        columns=["Predicted No Diabetes", "Predicted Diabetes"]
    )

    st.dataframe(cm, use_container_width=True)

    st.subheader("Selected Classification Threshold")

    st.write(
        "The project selected a probability threshold of 0.20 "
        "because it produced the highest F1-score among the "
        "tested thresholds."
    )

    threshold_table = pd.DataFrame({
        "Threshold": [0.10, 0.15, 0.20, 0.25, 0.30,
                      0.35, 0.40, 0.45, 0.50],

        "Precision": [
            0.272, 0.317, 0.360, 0.395, 0.430,
            0.464, 0.501, 0.539, 0.568
        ],

        "Recall": [
            0.854, 0.754, 0.659, 0.555, 0.462,
            0.374, 0.295, 0.226, 0.164
        ],

        "F1 Score": [
            0.413, 0.446, 0.466, 0.462, 0.446,
            0.414, 0.371, 0.318, 0.255
        ]
    })

    st.dataframe(
        threshold_table,
        use_container_width=True
    )


# ============================================================
# TAB 2 — PREDICTION
# ============================================================

with tab2:

    st.header("Diabetes Risk Prediction")

    st.write(
        "Enter the predictor values using the same coding structure "
        "used during model training."
    )

    st.warning(
        "The categorical variables below use the numerical codes "
        "from the BRFSS-derived dataset used in this project."
    )

    # --------------------------------------------------------
    # DEMOGRAPHIC VARIABLES
    # --------------------------------------------------------

    st.subheader("1. Demographic Information")

    col1, col2 = st.columns(2)

    with col1:

        sex_binary = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x:
                "Female (0)" if x == 0 else "Male (1)"
        )

        age_group = st.selectbox(
            "Age Group Code",
            options=[1, 2, 3, 4]
        )

        race = st.selectbox(
            "Race Code",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9]
        )

        education = st.selectbox(
            "Education Code",
            options=[1, 2, 3, 4]
        )

    with col2:

        income_group = st.selectbox(
            "Income Group Code",
            options=[1, 2, 3, 4, 5, 6, 7]
        )

        employment = st.selectbox(
            "Employment Code",
            options=[1, 2, 3, 4, 5, 6, 7, 8]
        )

        marital_status = st.selectbox(
            "Marital Status Code",
            options=[1, 2, 3, 4, 5, 6]
        )


    # --------------------------------------------------------
    # HEALTH VARIABLES
    # --------------------------------------------------------

    st.subheader("2. Health and Lifestyle Information")

    col1, col2 = st.columns(2)

    with col1:

        BMI = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=80.0,
            value=28.0,
            step=0.1
        )

        hypertension = st.selectbox(
            "Hypertension",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        high_cholesterol = st.selectbox(
            "High Cholesterol",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )

        general_health = st.selectbox(
            "General Health Code",
            options=[1, 2, 3, 4, 5]
        )

    with col2:

        physical_unhealthy_days = st.slider(
            "Physical Unhealthy Days",
            min_value=0,
            max_value=30,
            value=0
        )

        mental_unhealthy_days = st.slider(
            "Mental Unhealthy Days",
            min_value=0,
            max_value=30,
            value=0
        )

        exercise = st.selectbox(
            "Exercise",
            options=[0, 1],
            format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
        )


    # --------------------------------------------------------
    # DIET / BEHAVIOUR
    # --------------------------------------------------------

    st.subheader("3. Diet and Behaviour")

    col1, col2 = st.columns(2)

    with col1:

        fruit_times_day = st.number_input(
            "Fruit Consumption per Day",
            min_value=0.0,
            max_value=20.0,
            value=1.0,
            step=0.1
        )

        green_vegetable_times_day = st.number_input(
            "Green Vegetable Consumption per Day",
            min_value=0.0,
            max_value=20.0,
            value=0.5,
            step=0.1
        )

    with col2:

        drinks_per_week = st.number_input(
            "Drinks per Week",
            min_value=0.0,
            max_value=1000.0,
            value=0.0,
            step=1.0
        )

        smoker = st.selectbox(
            "Smoking Status Code",
            options=[1, 2, 3, 4, 9]
        )

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "sex_binary": [sex_binary],
        "age_group": [age_group],
        "race": [race],
        "education": [education],
        "income_group": [income_group],
        "employment": [employment],
        "marital_status": [marital_status],
        "BMI": [BMI],
        "hypertension": [hypertension],
        "high_cholesterol": [high_cholesterol],
        "general_health": [general_health],
        "physical_unhealthy_days": [physical_unhealthy_days],
        "mental_unhealthy_days": [mental_unhealthy_days],
        "exercise": [exercise],
        "fruit_times_day": [fruit_times_day],
        "green_vegetable_times_day": [green_vegetable_times_day],
        "drinks_per_week": [drinks_per_week],
        "_SMOKER3": [smoker]
    })


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    st.divider()

    predict_button = st.button(
        "🔮 Predict Diabetes Risk",
        type="primary"
    )

    if predict_button:

        probability = model.predict_proba(
            input_data
        )[0, 1]

        prediction = int(
            probability >= THRESHOLD
        )

        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Estimated Diabetes Probability",
                f"{probability * 100:.2f}%"
            )

        with result_col2:

            if prediction == 1:

                st.error(
                    "Higher predicted diabetes risk"
                )

            else:

                st.success(
                    "Lower predicted diabetes risk"
                )

        st.write(
            f"The model probability is compared with the "
            f"project threshold of {THRESHOLD:.2f}."
        )


# ============================================================
# TAB 3 — ABOUT
# ============================================================

with tab3:

    st.header("About This Project")

    st.write("""
    This project develops a diabetes prediction model using
    data from the CDC 2021 Behavioral Risk Factor Surveillance
    System (BRFSS).

    The analytical workflow includes:

    • Data extraction
    • Data cleaning
    • Feature engineering
    • Missing-value handling
    • Categorical encoding
    • Numerical standardization
    • Logistic regression
    • Model evaluation
    • Threshold optimization
    • Model persistence
    • Interactive prediction
    """)

    st.subheader("Model Inputs")

    predictors = [
        "sex_binary",
        "age_group",
        "race",
        "education",
        "income_group",
        "employment",
        "marital_status",
        "BMI",
        "hypertension",
        "high_cholesterol",
        "general_health",
        "physical_unhealthy_days",
        "mental_unhealthy_days",
        "exercise",
        "fruit_times_day",
        "green_vegetable_times_day",
        "drinks_per_week",
        "_SMOKER3"
    ]

    st.dataframe(
        pd.DataFrame({"Predictor": predictors}),
        use_container_width=True
    )

    st.caption(
        "This application is a project demonstration and "
        "should not be interpreted as a medical diagnostic system."
    )