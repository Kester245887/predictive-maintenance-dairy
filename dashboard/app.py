import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide"
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "random_forest_pipeline.joblib"
THRESHOLD_PATH = PROJECT_DIR / "models" / "decision_threshold.txt"


# --------------------------------------------------
# LOAD MODEL AND THRESHOLD
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

threshold = float(THRESHOLD_PATH.read_text().strip())

# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.title("🏭 Predictive Maintenance for Dairy Manufacturing")

st.markdown(
    """
    **A Data-Driven Approach to Reducing Equipment Downtime**

    This independent portfolio project demonstrates how machine-learning
    methods can support predictive maintenance decisions using the
    synthetic UCI AI4I 2020 dataset.

    *This is not an Arla Foods implementation and does not use Arla's internal data.*
    """
)

st.divider()


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Failure Prediction",
        "Model Performance",
        "Model Explainability",
        "Business Impact"
    ]
)

# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------

if page == "Overview":

    st.header("Project Overview")

    st.write(
        """
        The goal of this project is to demonstrate how machine-learning
        models can help identify equipment at risk of failure before
        breakdown occurs.

        The analysis uses the UCI AI4I 2020 Predictive Maintenance
        Dataset. The dataset is synthetic and represents generic
        industrial machinery rather than a real dairy production line.
        """
    )

    # Key project metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset records", "10,000")

    with col2:
        st.metric("Recorded failures", "339")

    with col3:
        st.metric("Failure rate", "3.39%")

    with col4:
        st.metric("Model", "Random Forest")

    st.divider()

    st.subheader("Selected Model Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Precision", "0.75")

    with col2:
        st.metric("Recall", "0.68")

    with col3:
        st.metric("F1-score", "0.71")

    with col4:
        st.metric("ROC-AUC", "0.965")

    st.caption(
        "Results are from the project's held-out test set using a "
        "decision threshold of 0.30. The test set was examined during "
        "project development, so these results should not be described "
        "as a completely untouched final evaluation."
    )
    
    # --------------------------------------------------
# FAILURE PREDICTION PAGE
# --------------------------------------------------

elif page == "Failure Prediction":

    st.header("Machine Failure Prediction")

    st.write(
        """
        Enter the machine's sensor readings below to estimate its
        probability of failure using the trained Random Forest model.

        This is a demonstration using synthetic data, not a live
        industrial monitoring system.
        """
    )

    st.subheader("Machine Sensor Readings")

    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"]
        )

        air_temperature = st.number_input(
            "Air Temperature (K)",
            min_value=295.0,
            max_value=304.5,
            value=300.0,
            step=0.1
        )

        process_temperature = st.number_input(
            "Process Temperature (K)",
            min_value=305.0,
            max_value=314.0,
            value=310.0,
            step=0.1
        )

    with col2:
        rotational_speed = st.number_input(
            "Rotational Speed (rpm)",
            min_value=1168,
            max_value=2886,
            value=1500,
            step=10
        )

        torque = st.number_input(
            "Torque (Nm)",
            min_value=3.8,
            max_value=76.6,
            value=40.0,
            step=0.1
        )

        tool_wear = st.number_input(
            "Tool Wear (min)",
            min_value=0,
            max_value=253,
            value=100,
            step=1
        )

    if st.button("Predict Failure Risk", type="primary"):

        input_data = pd.DataFrame(
            {
                "Type": [machine_type],
                "Air temperature [K]": [air_temperature],
                "Process temperature [K]": [process_temperature],
                "Rotational speed [rpm]": [rotational_speed],
                "Torque [Nm]": [torque],
                "Tool wear [min]": [tool_wear]
            }
        )

        failure_probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        st.metric(
            "Estimated Failure Probability",
            f"{failure_probability:.1%}"
        )

        if failure_probability >= threshold:
            st.error(
                "⚠️ The model flags this machine as potentially at risk. "
                "Consider reviewing it for maintenance."
            )
        else:
            st.success(
                "The model does not flag this machine as high risk "
                "at the selected decision threshold."
            )

        st.caption(
            f"Decision threshold: {threshold:.2f}. "
            "This is a model-generated estimate, not a guarantee "
            "that a machine will or will not fail."
        )
        
        # --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Model Performance")

    st.write(
        """
        The Random Forest model was evaluated on the project's
        held-out test set using a decision threshold of 0.30.
        """
    )

    st.subheader("Classification Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Precision", "0.75")

    with col2:
        st.metric("Recall", "0.68")

    with col3:
        st.metric("F1-score", "0.71")

    with col4:
        st.metric("ROC-AUC", "0.965")

    st.divider()

    st.subheader("Test Set Confusion Matrix")

    confusion_matrix = pd.DataFrame(
        [[1917, 15], [22, 46]],
        index=["Actual: No Failure", "Actual: Failure"],
        columns=["Predicted: No Failure", "Predicted: Failure"]
    )

    st.dataframe(confusion_matrix)

    st.write(
        """
        **Interpretation**

        - **1,917** machines were correctly classified as no failure.
        - **46** failures were correctly detected.
        - **22** failures were missed by the model.
        - **15** machines were flagged as potential failures but did not
          fail in the test set.
        """
    )

    st.caption(
        "The test set was examined during project development. "
        "These results should not be described as a completely "
        "untouched final evaluation."
    )
  

# --------------------------------------------------
# MODEL EXPLAINABILITY PAGE
# --------------------------------------------------

elif page == "Model Explainability":

    st.header("Model Explainability")

    st.write(
        """
        This page explains which equipment characteristics influence
        the Random Forest model's failure predictions.

        The analysis uses Random Forest feature importance and SHAP
        (SHapley Additive exPlanations) to improve model transparency.
        """
    )

    st.subheader("Random Forest Feature Importance")

    st.write(
        """
        The model's impurity-based feature importance identifies
        rotational speed, torque and tool wear as its three most
        influential predictors.
        """
    )

    st.subheader("SHAP Feature Importance")

    st.image(
        PROJECT_DIR / "reports" / "figures" / "shap_feature_importance.png",
        caption="Global SHAP feature importance",
        use_container_width=True
    )

    st.subheader("SHAP Beeswarm Plot")

    st.write(
        """
        The beeswarm plot shows how feature values influence model
        predictions across the sampled observations. Positive SHAP
        values push predictions towards failure, while negative values
        push them away from failure.
        """
    )

    st.image(
        PROJECT_DIR / "reports" / "figures" / "shap_beeswarm.png",
        caption="SHAP summary plot for failure predictions",
        use_container_width=True
    )

    st.subheader("Individual Prediction Explanation")

    st.write(
        """
        The waterfall plot explains an individual prediction for
        machine 4701, which was correctly classified as a non-failure.
        """
    )

    st.image(
        PROJECT_DIR / "reports" / "figures" / "shap_waterfall_machine_4701.png",
        caption="SHAP waterfall explanation for machine 4701",
        use_container_width=True
    )

    st.caption(
        "SHAP explanations describe the behaviour of this trained model. "
        "They do not establish causal relationships between sensor "
        "measurements and equipment failures."
    )
  
    # --------------------------------------------------
# BUSINESS IMPACT PAGE
# --------------------------------------------------

elif page == "Business Impact":

    st.header("Illustrative Business Impact")

    st.write(
        """
        This scenario estimates how predictive maintenance could affect
        costs if model-detected failures led to planned maintenance.

        The cost figures and prevention rates below are illustrative
        assumptions, not actual dairy-industry or Arla Foods benchmarks.
        """
    )

    st.subheader("Scenario Assumptions")

    col1, col2 = st.columns(2)

    with col1:
        unexpected_failure_cost = st.number_input(
            "Cost per unexpected failure (€)",
            min_value=0,
            value=5000,
            step=500
        )

        planned_maintenance_cost = st.number_input(
            "Cost per planned maintenance (€)",
            min_value=0,
            value=1000,
            step=100
        )

    with col2:
        false_alarm_cost = st.number_input(
            "Cost per false alarm (€)",
            min_value=0,
            value=200,
            step=50
        )

        prevention_rate = st.slider(
            "Assumed prevention rate for detected failures",
            min_value=0,
            max_value=100,
            value=70,
            step=10
        ) / 100

    # Test-set outcomes at the selected decision threshold
    true_positives = 46
    false_positives = 15
    false_negatives = 22

    # Baseline: all actual failures occur unexpectedly
    baseline_cost = (
        true_positives + false_negatives
    ) * unexpected_failure_cost

    # Scenario: detected failures receive planned maintenance
    planned_cost = (
        true_positives * planned_maintenance_cost
    )

    remaining_unexpected_failures = (
        true_positives * (1 - prevention_rate)
        + false_negatives
    )

    unexpected_cost = (
        remaining_unexpected_failures
        * unexpected_failure_cost
    )

    false_alarm_total = (
        false_positives * false_alarm_cost
    )

    scenario_cost = (
        planned_cost
        + unexpected_cost
        + false_alarm_total
    )

    estimated_savings = baseline_cost - scenario_cost

    savings_percentage = (
        estimated_savings / baseline_cost * 100
        if baseline_cost > 0
        else 0
    )

    st.divider()

    st.subheader("Estimated Scenario Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Baseline Cost",
            f"€{baseline_cost:,.0f}"
        )

    with col2:
        st.metric(
            "Scenario Cost",
            f"€{scenario_cost:,.0f}"
        )

    with col3:
        st.metric(
            "Estimated Savings",
            f"€{estimated_savings:,.0f}",
            f"{savings_percentage:.1f}%"
        )

    st.write("### Cost Breakdown")

    breakdown = pd.DataFrame(
        {
            "Cost Category": [
                "Planned maintenance",
                "Remaining unexpected failures",
                "False-alarm investigations"
            ],
            "Estimated Cost (€)": [
                planned_cost,
                unexpected_cost,
                false_alarm_total
            ]
        }
    )

    st.dataframe(
        breakdown,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Illustrative scenario only. It excludes implementation, "
        "monitoring, and other operating costs. Actual savings would "
        "depend on real equipment data and validated assumptions."
    )
