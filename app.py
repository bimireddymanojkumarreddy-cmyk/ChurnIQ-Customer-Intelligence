
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_DIR / "Data" /
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

MODEL_PATH = (
    PROJECT_DIR / "models" /
    "Customer_Churn_Model.pkl"
)

METRICS_PATH = (
    PROJECT_DIR / "reports" /
    "final_model_metrics.json"
)

SHAP_PATH = (
    PROJECT_DIR / "reports" /
    "shap_feature_importance.csv"
)

# Actual final test metrics from your model evaluation screenshot.
FALLBACK_METRICS = {
    "Accuracy": 0.7501774308019872,
    "Precision": 0.52,
    "Recall": 0.7647058823529411,
    "F1 Score": 0.6190476190476191,
    "ROC-AUC": 0.836813144230024
}

RAW_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

FEATURE_NAMES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
    "AverageChargesPerMonth", "TenureGroup",
    "MonthlyChargeGroup", "TotalAdditionalServices"
]

# =========================================================
# 2. CUSTOM STYLING
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #080D18;
        color: #F4F7FC;
    }

    [data-testid="stSidebar"] {
        background-color: #0E1727;
        border-right: 1px solid #24334A;
    }

    [data-testid="stSidebar"] * {
        color: #DCE6F5;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #F4F7FC;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #91A9C8;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .metric-card {
        background: #141F31;
        border: 1px solid #263852;
        border-radius: 14px;
        padding: 20px;
        min-height: 115px;
    }

    .metric-label {
        color: #A9BCD6;
        font-size: 14px;
    }

    .metric-value {
        color: #EAF1FC;
        font-size: 30px;
        font-weight: 800;
        margin-top: 7px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 12px;
        color: #F4F7FC;
    }

    .info-card {
        background: #141F31;
        border: 1px solid #263852;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 12px;
    }

    .workflow-step {
        background: #141F31;
        border-left: 4px solid #19C5B5;
        border-radius: 8px;
        padding: 15px 18px;
        margin-bottom: 12px;
    }

    .small-muted {
        color: #91A9C8;
        font-size: 13px;
    }

    h1, h2, h3 {
        color: #F4F7FC !important;
    }

    div[data-testid="stMetric"] {
        background: #141F31;
        border: 1px solid #263852;
        padding: 16px;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. LOAD DATA AND MODEL
# =========================================================

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    ).fillna(0)

    return data


@st.cache_resource
def load_model_package():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    df = load_dataset()
    model_package = load_model_package()

    model = model_package["model"]
    preprocessor = model_package["preprocessor"]
    threshold = float(model_package["threshold"])
    model_name = model_package["model_name"]
    saved_features = model_package["feature_names"]

except Exception as error:
    st.error(f"Application could not load: {error}")
    st.info(
        "Check that your dataset and saved model are in "
        "the correct project folders."
    )
    st.stop()


# =========================================================
# 4. FEATURE ENGINEERING
# =========================================================

def engineer_features(data):
    result = data.copy()

    result["TotalCharges"] = pd.to_numeric(
        result["TotalCharges"], errors="coerce"
    ).fillna(0)

    result["tenure"] = pd.to_numeric(
        result["tenure"], errors="coerce"
    ).fillna(0)

    result["MonthlyCharges"] = pd.to_numeric(
        result["MonthlyCharges"], errors="coerce"
    ).fillna(0)

    result["AverageChargesPerMonth"] = (
        result["TotalCharges"] /
        result["tenure"].replace(0, 1)
    )

    result["TenureGroup"] = pd.cut(
        result["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "New",
            "Short-term",
            "Medium-term",
            "Long-term"
        ]
    )

    result["MonthlyChargeGroup"] = pd.cut(
        result["MonthlyCharges"],
        bins=[0, 35, 65, 100, float("inf")],
        labels=[
            "Low",
            "Medium",
            "High",
            "Very High"
        ],
        include_lowest=True
    )

    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    result["TotalAdditionalServices"] = (
        result[service_columns] == "Yes"
    ).sum(axis=1)

    return result


# =========================================================
# 5. PREDICTION FUNCTIONS
# =========================================================

def prepare_model_input(data):
    engineered = engineer_features(data)

    missing = [
        column for column in saved_features
        if column not in engineered.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    return engineered[saved_features]


def predict_customers(data):
    model_input = prepare_model_input(data)

    transformed = preprocessor.transform(model_input)

    probabilities = model.predict_proba(transformed)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    output = data.copy()

    output["Churn Probability"] = probabilities
    output["Predicted Churn"] = np.where(
        predictions == 1, "Yes", "No"
    )

    output["Risk Level"] = [
        "High" if p >= threshold
        else "Medium" if p >= 0.30
        else "Low"
        for p in probabilities
    ]

    return output


# =========================================================
# 6. RETENTION RECOMMENDATIONS
# =========================================================

def get_recommendations(customer):
    recommendations = []

    if customer["Contract"] == "Month-to-month":
        recommendations.append(
            "Offer a discounted one-year contract "
            "to encourage long-term retention."
        )

    if customer["tenure"] <= 12:
        recommendations.append(
            "Provide an onboarding experience and "
            "an early-stage loyalty offer."
        )

    if customer["MonthlyCharges"] >= 70:
        recommendations.append(
            "Review the customer's bill and offer "
            "a suitable pricing plan."
        )

    if customer["OnlineSecurity"] == "No":
        recommendations.append(
            "Offer an Online Security package or "
            "an introductory discount."
        )

    if customer["TechSupport"] == "No":
        recommendations.append(
            "Consider offering a technical support "
            "trial or discounted support plan."
        )

    if customer["PaymentMethod"] == "Electronic check":
        recommendations.append(
            "Encourage automatic payment methods "
            "with a convenient payment offer."
        )

    if not recommendations:
        recommendations.append(
            "Continue regular engagement and monitor "
            "customer satisfaction."
        )

    return recommendations


# =========================================================
# 7. LOAD MODEL METRICS
# =========================================================

def load_metrics():
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as file:
                content = json.load(file)

            return content.get(
                "test_metrics", FALLBACK_METRICS
            )
        except Exception:
            return FALLBACK_METRICS

    return FALLBACK_METRICS


metrics = load_metrics()


# =========================================================
# 8. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown(
        "<h2 style='color:#19C5B5;'>◉ ChurnIQ</h2>",
        unsafe_allow_html=True
    )

    st.caption("CUSTOMER INTELLIGENCE PLATFORM")

    st.divider()

    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Churn Analysis",
            "Customer Prediction",
            "Batch Prediction",
            "Retention Analytics",
            "Model Performance",
            "About Project"
        ]
    )

    st.divider()

    st.markdown("### ACTIVE MODEL")
    st.markdown(f"**{model_name}**")
    st.caption(f"Decision threshold: {threshold:.2f}")
    st.caption(f"Input features: {len(saved_features)}")

    st.divider()

    st.caption(
        "Customer Churn Prediction & Retention Analytics"
    )


# =========================================================
# 9. COMMON DATA PREPARATION
# =========================================================

total_customers = len(df)
churned_customers = int((df["Churn"] == "Yes").sum())
churn_rate = (
    churned_customers / total_customers * 100
)
average_charges = df["MonthlyCharges"].mean()

df_engineered = engineer_features(df)


def page_heading(title, subtitle):
    st.markdown(
        f"<div class='main-title'>{title}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='subtitle'>{subtitle}</div>",
        unsafe_allow_html=True
    )


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 10. OVERVIEW DASHBOARD
# =========================================================

if page == "Overview":

    page_heading(
        "Customer Overview",
        "Monitor customer health, churn patterns and retention opportunities."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Total Customers",
            f"{total_customers:,}"
        )

    with c2:
        metric_card(
            "Churned Customers",
            f"{churned_customers:,}"
        )

    with c3:
        metric_card(
            "Actual Churn Rate",
            f"{churn_rate:.2f}%"
        )

    with c4:
        metric_card(
            "Avg. Monthly Charges",
            f"${average_charges:.2f}"
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown(
            "### Customer Churn Distribution"
        )

        churn_counts = (
            df["Churn"]
            .value_counts()
            .reindex(["No", "Yes"])
        )

        fig = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            hole=0.65,
            color=churn_counts.index,
            color_discrete_map={
                "No": "#19C5B5",
                "Yes": "#B66AD8"
            }
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            textfont_size=12,
            marker=dict(
                line=dict(
                    color="#141F31",
                    width=2
                )
            )
        )

        fig.update_layout(
            height=330,
            margin=dict(
                l=10, r=10, t=10, b=25
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.08,
                xanchor="center",
                x=0.5
            ),
            paper_bgcolor="#141F31",
            plot_bgcolor="#141F31",
            font_color="#EAF1FC"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    with right:
        st.markdown("### Churn by Contract Type")

        contract_churn = (
            df.groupby("Contract", observed=False)
            .apply(
                lambda x: (x["Churn"] == "Yes").mean() * 100,
                include_groups=False
            )
            .reset_index(name="Churn Rate")
        )

        fig = px.bar(
            contract_churn,
            x="Contract",
            y="Churn Rate",
            color="Contract",
            text="Churn Rate",
            color_discrete_sequence=[
                "#19C5B5",
                "#B66AD8",
                "#5B9CF5"
            ]
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            height=330,
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="#141F31",
            plot_bgcolor="#141F31",
            font_color="#EAF1FC",
            yaxis_title="Churn Rate (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    left, right = st.columns(2)

    with left:
        st.markdown("### Churn by Tenure Group")

        tenure_churn = (
            df_engineered.groupby(
                "TenureGroup", observed=False
            )["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
        )

        fig = px.bar(
            tenure_churn,
            x="TenureGroup",
            y="Churn Rate",
            color="Churn Rate",
            color_continuous_scale="Teal",
            text="Churn Rate"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            height=350,
            paper_bgcolor="#141F31",
            plot_bgcolor="#141F31",
            font_color="#EAF1FC"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:
        st.markdown("### Average Monthly Charges")

        charge_data = (
            df.groupby("Churn")["MonthlyCharges"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            charge_data,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map={
                "No": "#19C5B5",
                "Yes": "#B66AD8"
            },
            text="MonthlyCharges"
        )

        fig.update_traces(
            texttemplate="$%{text:.2f}",
            textposition="outside"
        )

        fig.update_layout(
            height=350,
            showlegend=False,
            paper_bgcolor="#141F31",
            plot_bgcolor="#141F31",
            font_color="#EAF1FC"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# 11. CHURN ANALYSIS
# =========================================================

elif page == "Churn Analysis":

    page_heading(
        "Churn Analysis",
        "Explore the customer characteristics associated with churn."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Churn by Internet Service")

        internet_data = (
            df.groupby("InternetService")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
        )

        fig = px.bar(
            internet_data,
            x="InternetService",
            y="Churn Rate",
            color="Churn Rate",
            color_continuous_scale="Teal",
            text="Churn Rate"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            height=350,
            paper_bgcolor="#141F31",
            plot_bgcolor="#141F31",
            font_color="#EAF1FC"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Churn by Payment Method")

        payment_data = (
            df.groupby("PaymentMethod")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
        )

        fig = px.bar(
            payment_data,
            x="PaymentMethod",
            y="Churn Rate",
            color="Churn Rate",
            color_continuous_scale="Teal",
            text="Churn Rate"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            height=350,
            paper_bgcolor="#141F31",
            plot_bgcolor="#141F31",
            font_color="#EAF1FC",
            xaxis_tickangle=-25
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Churn Rate by Customer Characteristics")

    selected_column = st.selectbox(
        "Choose a customer characteristic",
        [
            "Contract",
            "InternetService",
            "PaymentMethod",
            "OnlineSecurity",
            "TechSupport",
            "SeniorCitizen",
            "PaperlessBilling"
        ]
    )

    analysis = (
        df.groupby(selected_column, dropna=False)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Churn Rate")
    )

    st.dataframe(
        analysis.sort_values(
            "Churn Rate", ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 12. CUSTOMER PREDICTION
# =========================================================

elif page == "Customer Prediction":

    page_heading(
        "Customer Churn Prediction",
        "Estimate churn probability and generate retention recommendations."
    )

    with st.form("customer_prediction_form"):

        st.markdown("### Customer Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender", ["Male", "Female"]
            )

            senior = st.selectbox(
                "Senior Citizen", [0, 1]
            )

            partner = st.selectbox(
                "Partner", ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents", ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12
            )

            phone = st.selectbox(
                "Phone Service", ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

        with col2:
            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        with col3:
            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=2000.0,
                value=70.0
            )

            total = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=100000.0,
                value=840.0
            )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            use_container_width=True
        )

    if submitted:

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        try:
            result = predict_customers(customer)

            probability = float(
                result["Churn Probability"].iloc[0]
            )

            prediction = result["Predicted Churn"].iloc[0]
            risk = result["Risk Level"].iloc[0]

            st.divider()
            st.markdown("## Prediction Results")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )

            with c2:
                st.metric(
                    "Prediction",
                    "Likely to Churn"
                    if prediction == "Yes"
                    else "Likely to Stay"
                )

            with c3:
                st.metric("Risk Level", risk)

            st.progress(
                min(max(probability, 0.0), 1.0)
            )

            if prediction == "Yes":
                st.warning(
                    "The model predicts that this customer "
                    "is likely to churn."
                )
            else:
                st.success(
                    "The model predicts that this customer "
                    "is likely to stay."
                )

            st.markdown("### Personalized Retention Recommendations")

            for recommendation in get_recommendations(
                customer.iloc[0]
            ):
                st.markdown(
                    f"- {recommendation}"
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# =========================================================
# 13. BATCH PREDICTION
# =========================================================

elif page == "Batch Prediction":

    page_heading(
        "Batch Customer Prediction",
        "Upload a customer CSV file to predict churn for multiple customers."
    )

    st.info(
        "Upload a CSV containing the original customer columns "
        "used by this project. The Churn column is not required."
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            batch_df = pd.read_csv(uploaded_file)

            st.markdown("### Uploaded Data Preview")

            st.dataframe(
                batch_df.head(10),
                use_container_width=True
            )

            missing_columns = [
                column for column in RAW_FEATURES
                if column not in batch_df.columns
            ]

            if missing_columns:
                st.error(
                    f"Missing required columns: {missing_columns}"
                )

            else:
                if st.button(
                    "Run Batch Predictions",
                    use_container_width=True
                ):

                    results = predict_customers(batch_df)

                    st.success(
                        "Batch predictions completed!"
                    )

                    total = len(results)

                    predicted_churn = int(
                        (
                            results["Predicted Churn"] == "Yes"
                        ).sum()
                    )

                    predicted_rate = (
                        predicted_churn / total * 100
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric("Total Customers", f"{total:,}")
                    c2.metric("Predicted Churn", f"{predicted_churn:,}")
                    c3.metric(
                        "Predicted Churn Rate",
                        f"{predicted_rate:.2f}%"
                    )

                    st.markdown("### Prediction Results")

                    st.dataframe(
                        results,
                        use_container_width=True,
                        height=400
                    )

                    csv_data = results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "Download Predictions CSV",
                        data=csv_data,
                        file_name="customer_churn_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as error:
            st.error(f"Batch prediction failed: {error}")


# =========================================================
# 14. RETENTION ANALYTICS
# =========================================================

elif page == "Retention Analytics":

    page_heading(
        "Retention Analytics",
        "Identify customer segments with higher churn and retention opportunities."
    )

    tenure_analysis = (
        df_engineered.groupby(
            "TenureGroup", observed=False
        )
        .agg(
            Total_Customers=("Churn", "size"),
            Churned_Customers=(
                "Churn",
                lambda x: (x == "Yes").sum()
            )
        )
        .reset_index()
    )

    tenure_analysis["Churn Rate"] = (
        tenure_analysis["Churned_Customers"] /
        tenure_analysis["Total_Customers"] * 100
    )

    st.markdown("### Customer Retention by Tenure")

    st.dataframe(
        tenure_analysis,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        tenure_analysis,
        x="TenureGroup",
        y="Churn Rate",
        color="Churn Rate",
        color_continuous_scale="Teal",
        text="Churn Rate"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        height=400,
        paper_bgcolor="#141F31",
        plot_bgcolor="#141F31",
        font_color="#EAF1FC"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("### Retention Opportunities")

    highest = tenure_analysis.loc[
        tenure_analysis["Churn Rate"].idxmax()
    ]

    lowest = tenure_analysis.loc[
        tenure_analysis["Churn Rate"].idxmin()
    ]

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>Highest Churn Segment</h3>
                <h2 style="color:#B66AD8;">
                    {highest['TenureGroup']}
                </h2>
                <p>
                    Churn rate:
                    <b>{highest['Churn Rate']:.2f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>Lowest Churn Segment</h3>
                <h2 style="color:#19C5B5;">
                    {lowest['TenureGroup']}
                </h2>
                <p>
                    Churn rate:
                    <b>{lowest['Churn Rate']:.2f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Suggested Retention Actions")

    st.markdown("""
    - Focus early engagement on new customers.
    - Encourage month-to-month customers to consider longer contracts.
    - Review high monthly charges and suitable pricing options.
    - Improve access to online security and technical support.
    - Monitor high-risk customers using model predictions.
    """)


# =========================================================
# 15. MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    page_heading(
        "Model Performance & Insights",
        "Review the selected model, decision threshold and final test results."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Selected Model", model_name)
    c2.metric("Decision Threshold", f"{threshold:.2f}")
    c3.metric("Input Features", len(saved_features))

    st.divider()

    st.markdown("### Final Test Performance")

    metric_columns = st.columns(5)

    for column, (name, value) in zip(
        metric_columns, metrics.items()
    ):
        try:
            column.metric(
                name,
                f"{float(value) * 100:.2f}%"
            )
        except (ValueError, TypeError):
            column.metric(name, str(value))

    metric_df = pd.DataFrame(
        [
            {
                "Metric": name,
                "Score": float(value)
            }
            for name, value in metrics.items()
            if isinstance(value, (int, float))
        ]
    )

    if not metric_df.empty:
        fig = px.bar(
            metric_df,
            x="Metric",
            y="Score",
            color="Metric",
            text="Score",
            color_discrete_sequence=[
                "#19C5B5",
                "#B66AD8",
                "#5B9CF5",
                "#E9A23B",
                "#67C587"
            ]
        )

        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

        fig.update_layout(
            height=400,
            showlegend=False,
            yaxis_range=[0, 1],
            paper_bgcolor="#141F31",
            plot_bgcolor="#141F31",
            font_color="#EAF1FC"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.caption(
        "Metrics are from the final test evaluation. "
        "The threshold controls the predicted churn classification."
    )

    st.divider()

    st.markdown("### Feature Importance")

    if SHAP_PATH.exists():

        try:
            shap_df = pd.read_csv(SHAP_PATH)

            st.dataframe(
                shap_df,
                use_container_width=True
            )

            numeric_columns = shap_df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if len(numeric_columns) > 0:
                value_column = numeric_columns[-1]

                name_column = next(
                    (
                        col for col in shap_df.columns
                        if col not in numeric_columns
                    ),
                    None
                )

                if name_column:
                    importance_fig = px.bar(
                        shap_df.nlargest(
                            15, value_column
                        ),
                        x=value_column,
                        y=name_column,
                        orientation="h",
                        color=value_column,
                        color_continuous_scale="Teal"
                    )

                    importance_fig.update_layout(
                        height=500,
                        paper_bgcolor="#141F31",
                        plot_bgcolor="#141F31",
                        font_color="#EAF1FC"
                    )

                    st.plotly_chart(
                        importance_fig,
                        use_container_width=True
                    )

        except Exception as error:
            st.warning(
                f"Could not display feature importance: {error}"
            )

    else:
        st.info(
            "SHAP feature importance file is not available yet. "
            "The rest of the model performance section is available."
        )


# =========================================================
# 16. ABOUT PROJECT AND PROJECT WORKFLOW
# =========================================================
elif page == "About Project":

    st.title("About the Project")
    st.caption(
        "Customer Churn Prediction & Retention Analytics Platform"
    )

    # --------------------------------------------------
    # PROJECT OVERVIEW
    # --------------------------------------------------

    st.markdown("## Project Overview")

    st.markdown("""
    This project is an end-to-end Machine Learning and Customer
    Analytics Platform designed to identify customers who are
    likely to discontinue a telecommunications service.

    It combines customer data analysis, feature engineering,
    machine learning, churn probability estimation, and
    personalized retention recommendations.

    The platform helps businesses understand customer behavior,
    identify customers at risk of leaving, and make data-driven
    retention decisions.
    """)

    st.divider()

    # --------------------------------------------------
    # PROJECT OBJECTIVES
    # --------------------------------------------------

    st.markdown("## Project Objectives")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - Analyze customer demographics and subscription patterns.
        - Identify important factors associated with churn.
        - Develop a machine learning model to predict churn.
        - Estimate individual customer churn probabilities.
        """)

    with col2:
        st.markdown("""
        - Classify customers into risk categories.
        - Generate personalized retention recommendations.
        - Support batch customer predictions.
        - Present business insights through an interactive dashboard.
        """)

    st.divider()

    # --------------------------------------------------
    # PROJECT WORKFLOW
    # --------------------------------------------------

    st.markdown("## Project Workflow")

    st.caption(
        "End-to-end workflow followed to develop the platform."
    )

    workflow = [
        (
            "01",
            "Data Collection",
            "Used the IBM Telco Customer Churn dataset containing "
            "customer, subscription, service, and billing information."
        ),
        (
            "02",
            "Data Cleaning",
            "Handled missing values, converted data types, "
            "removed duplicate records, and prepared the dataset."
        ),
        (
            "03",
            "Exploratory Data Analysis",
            "Analyzed churn distribution, customer tenure, "
            "contract types, and monthly charges."
        ),
        (
            "04",
            "Feature Engineering",
            "Created AverageChargesPerMonth, TenureGroup, "
            "MonthlyChargeGroup, and TotalAdditionalServices."
        ),
        (
            "05",
            "Data Preprocessing",
            "Applied numerical and categorical preprocessing "
            "using imputation, scaling, and one-hot encoding."
        ),
        (
            "06",
            "Model Development",
            "Developed and evaluated machine learning models "
            "and selected Random Forest for this application."
        ),
        (
            "07",
            "Model Evaluation",
            "Evaluated the selected model on the test dataset "
            "using accuracy, precision, recall, F1-score, and ROC-AUC."
        ),
        (
            "08",
            "Model Deployment",
            "Saved the trained model and preprocessing pipeline "
            "for reuse in the Streamlit application."
        ),
        (
            "09",
            "Prediction & Retention",
            "Implemented individual predictions, batch predictions, "
            "churn risk classification, and retention recommendations."
        ),
        (
            "10",
            "Interactive Dashboard",
            "Built an interactive dashboard to explore customer "
            "analytics, predictions, and model performance."
        )
    ]

    for number, title, description in workflow:

        with st.container(border=True):
            col1, col2 = st.columns([0.12, 0.88])

            with col1:
                st.markdown(f"### {number}")

            with col2:
                st.markdown(f"### {title}")
                st.write(description)

    st.divider()

    # --------------------------------------------------
    # TECHNOLOGY STACK
    # --------------------------------------------------

    st.markdown("## Technology Stack")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:
        st.markdown("### Programming")
        st.markdown("""
        - Python
        - Pandas
        - NumPy
        """)

    with tech_col2:
        st.markdown("### Machine Learning")
        st.markdown("""
        - Scikit-learn
        - Random Forest
        - XGBoost
        - TensorFlow / Keras
        - SHAP
        """)

    with tech_col3:
        st.markdown("### Application")
        st.markdown("""
        - Streamlit
        - Plotly
        - Joblib
        - Git / GitHub
        """)

    st.divider()

    # --------------------------------------------------
    # APPLICATION FEATURES
    # --------------------------------------------------

    st.markdown("## Key Application Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.markdown("""
        **Customer Analytics**
        - Customer overview dashboard
        - Churn distribution analysis
        - Contract-based churn analysis
        - Tenure-based customer segmentation

        **Customer Prediction**
        - Individual customer prediction
        - Churn probability estimation
        - Low, Medium, and High risk categories
        """)

    with feature_col2:
        st.markdown("""
        **Retention Analytics**
        - Personalized retention recommendations
        - Customer risk identification
        - Business-focused insights

        **Model Insights**
        - Model performance metrics
        - Decision threshold information
        - Feature importance, when available
        - Batch prediction and CSV download
        """)

    st.divider()

    # --------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------

    st.markdown("## Model Information")

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:
        st.metric("Selected Model", model_package["model_name"])

    with model_col2:
        st.metric(
            "Decision Threshold",
            f'{model_package["threshold"]:.2f}'
        )

    with model_col3:
        st.metric(
            "Input Features",
            len(model_package["feature_names"])
        )

    st.info(
        "The application uses a saved model package containing "
        "the trained estimator, preprocessing pipeline, "
        "decision threshold, and expected input features."
    )

    st.divider()

    # --------------------------------------------------
    # BUSINESS VALUE
    # --------------------------------------------------

    st.markdown("## Business Value")

    st.markdown("""
    This platform demonstrates how machine learning can support
    customer retention strategies.

    By identifying customers with a higher estimated probability
    of churn, businesses can prioritize customer engagement,
    review subscription plans, and develop targeted retention
    campaigns.

    The objective is to transform customer data into actionable
    insights that support better business decisions.
    """)

    st.divider()

    # --------------------------------------------------
    # PROJECT SUMMARY
    # --------------------------------------------------

    st.markdown("## Project Summary")

    st.markdown("""
    **Project:** Customer Churn Prediction & Retention Analytics

    **Dataset:** IBM Telco Customer Churn

    **Application:** Interactive Streamlit Dashboard

    **Machine Learning Model:** Random Forest

    **Project Type:** End-to-End Machine Learning & Data Analytics
    """)