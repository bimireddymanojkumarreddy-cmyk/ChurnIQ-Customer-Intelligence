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
# 2. CUSTOM STYLING - CLEAN LIGHT THEME
# =========================================================

st.markdown("""
<style>
:root {
    --ink: #172B4D;
    --muted: #667085;
    --teal: #0F9D8A;
    --teal-dark: #087F70;
    --line: #E6EBF2;
    --surface: #FFFFFF;
    --canvas: #F5F8FC;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--canvas) !important;
    color: var(--ink) !important;
}
.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
[data-testid="stHeader"] {
    background: rgba(245,248,252,.92);
}
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.4rem 1rem;
}
[data-testid="stSidebar"] * { color: var(--ink); }
[data-testid="stSidebar"] h2 { color: var(--teal) !important; letter-spacing: -.5px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: .42rem .55rem;
    border-radius: 9px;
    font-weight: 550;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #EAF7F5;
}
h1, h2, h3, h4, h5, h6 {
    color: var(--ink) !important;
    letter-spacing: -.35px;
}
.main-title {
    font-size: clamp(1.8rem, 3vw, 2.45rem);
    font-weight: 800;
    letter-spacing: -1.1px;
    color: var(--ink) !important;
    margin: 0 0 .25rem 0;
}
.subtitle {
    color: var(--muted) !important;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.15rem 1.25rem;
    min-height: 118px;
    box-shadow: 0 5px 18px rgba(23,43,77,.045);
    transition: transform .18s ease, box-shadow .18s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 9px 24px rgba(23,43,77,.08);
}
.metric-label {
    color: var(--muted) !important;
    font-size: .83rem;
    font-weight: 600;
}
.metric-value {
    color: var(--ink) !important;
    font-size: clamp(1.55rem, 2.4vw, 2rem);
    font-weight: 800;
    letter-spacing: -.7px;
    margin-top: .45rem;
}
.section-title {
    font-size: 1.35rem;
    font-weight: 750;
    margin: 1.3rem 0 .75rem;
    color: var(--ink) !important;
}
.info-card, [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
}
.info-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    padding: 1.2rem;
    margin-bottom: .8rem;
}
.workflow-step {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-left: 4px solid var(--teal);
    border-radius: 12px;
    padding: 1rem 1.15rem;
    margin-bottom: .75rem;
}
.small-muted { color: var(--muted) !important; font-size: .82rem; }
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    padding: 1rem 1.1rem;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(23,43,77,.035);
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 750; }
input, textarea, [data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: var(--ink) !important;
    border-color: #D5DDE8 !important;
    border-radius: 9px !important;
}
[data-testid="stNumberInput"] button {
    color: var(--ink) !important;
}
.stButton button, .stFormSubmitButton button, .stDownloadButton button {
    background: var(--teal) !important;
    color: #FFFFFF !important;
    border: 0 !important;
    border-radius: 10px !important;
    min-height: 2.65rem;
    font-weight: 650;
    transition: background .15s ease, transform .15s ease;
}
.stButton button:hover, .stFormSubmitButton button:hover,
.stDownloadButton button:hover {
    background: var(--teal-dark) !important;
    color: #FFFFFF !important;
    transform: translateY(-1px);
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stTabs"] button { color: var(--ink) !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--teal) !important;
    border-bottom-color: var(--teal) !important;
}
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
}
[data-testid="stAlert"] { border-radius: 12px; }
hr { border-color: var(--line) !important; }
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
        "<h2 style='color:#0F9D8A; font-size:1.75rem; margin-bottom:0;'>◉ ChurnIQ</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style=\"font-size:.67rem; letter-spacing:1.15px; font-weight:700; color:#8290A5; margin-top:2px;\">CUSTOMER INTELLIGENCE PLATFORM</div>",
        unsafe_allow_html=True
    )

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

churned_customers = int(
    (df["Churn"] == "Yes").sum()
)

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
                    color="#FFFFFF",
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
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#172B4D",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(
                showgrid=True,
                gridcolor="#E9EEF5",
                zeroline=False
            )
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
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#172B4D",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(
                showgrid=True,
                gridcolor="#E9EEF5",
                zeroline=False
            ),
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
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#172B4D",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(
                showgrid=True,
                gridcolor="#E9EEF5",
                zeroline=False
            )
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
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#172B4D",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(
                showgrid=True,
                gridcolor="#E9EEF5",
                zeroline=False
            )
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
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#172B4D",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#E9EEF5", zeroline=False)
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
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#172B4D",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#E9EEF5", zeroline=False),
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
        "Identify high-risk customer segments and explore opportunities "
        "to improve customer retention."
    )

    # -----------------------------------------------------
    # RETENTION KPI METRICS
    # -----------------------------------------------------

    retained_customers = int(
        (df["Churn"] == "No").sum()
    )

    retention_rate = (
        retained_customers / total_customers * 100
        if total_customers > 0 else 0
    )

    churned_revenue = df.loc[
        df["Churn"] == "Yes",
        "MonthlyCharges"
    ].sum()

    avg_tenure = df["tenure"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Retained Customers",
            f"{retained_customers:,}"
        )

    with col2:
        metric_card(
            "Retention Rate",
            f"{retention_rate:.2f}%"
        )

    with col3:
        metric_card(
            "At-Risk Monthly Revenue",
            f"${churned_revenue:,.2f}"
        )

    with col4:
        metric_card(
            "Average Customer Tenure",
            f"{avg_tenure:.1f} months"
        )

    st.divider()

    # -----------------------------------------------------
    # CHURN RISK SEGMENTS
    # -----------------------------------------------------

    st.markdown("### Customer Retention Opportunities")

    left, right = st.columns(2)

    with left:

        st.markdown("#### Churn by Contract Type")

        contract_retention = (
            df.groupby("Contract")["Churn"]
            .apply(
                lambda x: (x == "Yes").mean() * 100
            )
            .reset_index(name="Churn Rate")
            .sort_values("Churn Rate", ascending=False)
        )

        fig = px.bar(
            contract_retention,
            x="Contract",
            y="Churn Rate",
            color="Churn Rate",
            text="Churn Rate",
            color_continuous_scale="Teal"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            height=350,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#172B4D",
            xaxis_title="Contract Type",
            yaxis_title="Churn Rate (%)",
            xaxis=dict(showgrid=False),
            yaxis=dict(
                showgrid=True,
                gridcolor="#E9EEF5"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.markdown("#### Churn by Tenure Group")

        tenure_retention = (
            df_engineered.groupby(
                "TenureGroup",
                observed=False
            )["Churn"]
            .apply(
                lambda x: (x == "Yes").mean() * 100
            )
            .reset_index(name="Churn Rate")
            .sort_values("Churn Rate", ascending=False)
        )

        fig = px.bar(
            tenure_retention,
            x="TenureGroup",
            y="Churn Rate",
            color="Churn Rate",
            text="Churn Rate",
            color_continuous_scale="Teal"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            height=350,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font_color="#172B4D",
            xaxis_title="Tenure Group",
            yaxis_title="Churn Rate (%)",
            xaxis=dict(showgrid=False),
            yaxis=dict(
                showgrid=True,
                gridcolor="#E9EEF5"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # -----------------------------------------------------
    # REVENUE AT RISK
    # -----------------------------------------------------

    st.markdown("### Monthly Revenue at Risk")

    revenue_data = (
        df.groupby("Churn")["MonthlyCharges"]
        .sum()
        .reset_index(name="Monthly Revenue")
    )

    fig = px.bar(
        revenue_data,
        x="Churn",
        y="Monthly Revenue",
        color="Churn",
        text="Monthly Revenue",
        color_discrete_map={
            "No": "#19C5B5",
            "Yes": "#B66AD8"
        }
    )

    fig.update_traces(
        texttemplate="$%{text:,.0f}",
        textposition="outside"
    )

    fig.update_layout(
        height=380,
        showlegend=False,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font_color="#172B4D",
        xaxis_title="Customer Status",
        yaxis_title="Total Monthly Charges ($)",
        xaxis=dict(showgrid=False),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E9EEF5"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -----------------------------------------------------
    # RETENTION ACTION INSIGHTS
    # -----------------------------------------------------

    st.markdown("### Recommended Retention Actions")

    highest_contract = contract_retention.iloc[0]

    highest_tenure = tenure_retention.iloc[0]

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Highest-Risk Contract</div>
                <div class="metric-value">
                    {highest_contract["Contract"]}
                </div>
                <div class="metric-label">
                    Churn rate: {highest_contract["Churn Rate"]:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with insight_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Highest-Risk Tenure Group</div>
                <div class="metric-value">
                    {highest_tenure["TenureGroup"]}
                </div>
                <div class="metric-label">
                    Churn rate: {highest_tenure["Churn Rate"]:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "Retention strategy: prioritize customers in the highest-risk "
        "segments. Consider personalized offers, contract incentives, "
        "service improvements and proactive customer support."
    )

# =========================================================
# 15. MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    page_heading(
        "Model Performance",
        "Evaluate model performance, compare classification algorithms "
        "and understand prediction quality."
    )

    # -----------------------------------------------------
    # MODEL INFORMATION
    # -----------------------------------------------------

    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Selected Model", model_name)

    with col2:
        metric_card("Decision Threshold", f"{threshold:.2f}")

    with col3:
        metric_card("Input Features", str(len(saved_features)))

    st.divider()

    # -----------------------------------------------------
    # MODEL COMPARISON
    # -----------------------------------------------------

    st.subheader("Model Comparison")

    model_results = pd.DataFrame({
        "Model": [
            "Random Forest",
            "XGBoost",
            "Logistic Regression",
            "ANN",
            "Decision Tree"
        ],
        "Accuracy": [
            0.772, 0.762, 0.752, 0.815, 0.731
        ],
        "Precision": [
            0.551, 0.536, 0.522, 0.686, 0.496
        ],
        "Recall": [
            0.759, 0.769, 0.779, 0.562, 0.773
        ],
        "F1 Score": [
            0.639, 0.632, 0.626, 0.618, 0.604
        ],
        "ROC-AUC": [
            0.847, 0.844, 0.850, 0.848, 0.818
        ]
    })

    st.dataframe(
        model_results.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1 Score": "{:.3f}",
            "ROC-AUC": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Comparison of the five evaluated classification models "
        "using the supplied evaluation results."
    )

    # -----------------------------------------------------
    # PERFORMANCE VISUALIZATION
    # -----------------------------------------------------

    st.markdown("### Performance Metric Comparison")

    selected_metric = st.selectbox(
        "Select Evaluation Metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    )

    fig = px.bar(
        model_results,
        x="Model",
        y=selected_metric,
        color="Model",
        text=selected_metric,
        color_discrete_sequence=[
            "#19C5B5",
            "#5B9CF5",
            "#B66AD8",
            "#F4A261",
            "#8290A5"
        ]
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside"
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font_color="#172B4D",
        yaxis=dict(
            title=selected_metric,
            range=[0, 1],
            gridcolor="#E9EEF5"
        ),
        xaxis=dict(
            title="Classification Model",
            showgrid=False
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------------
    # XGBOOST CLASSIFICATION REPORT
    # -----------------------------------------------------

    st.subheader("XGBoost Classification Report")

    st.caption(
        "Detailed class-wise evaluation results for the supplied "
        "XGBoost classification report."
    )

    classification_report_df = pd.DataFrame({
        "Class": [
            "No Churn",
            "Churn",
            "Macro Average",
            "Weighted Average"
        ],
        "Precision": [
            0.90, 0.52, 0.71, 0.80
        ],
        "Recall": [
            0.74, 0.76, 0.75, 0.75
        ],
        "F1 Score": [
            0.81, 0.62, 0.72, 0.76
        ],
        "Support": [
            1035, 374, 1409, 1409
        ]
    })

    st.dataframe(
        classification_report_df,
        use_container_width=True,
        hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:
        metric_card(
            "XGBoost Accuracy",
            "75.02%"
        )

    with col2:
        metric_card(
            "XGBoost ROC-AUC",
            "0.8368"
        )

    st.divider()

    # -----------------------------------------------------
    # MODEL PERFORMANCE INSIGHTS
    # -----------------------------------------------------

    st.subheader("Performance Insights")

    best_accuracy = model_results.loc[
        model_results["Accuracy"].idxmax()
    ]

    best_precision = model_results.loc[
        model_results["Precision"].idxmax()
    ]

    best_recall = model_results.loc[
        model_results["Recall"].idxmax()
    ]

    best_roc = model_results.loc[
        model_results["ROC-AUC"].idxmax()
    ]

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:

        st.info(
            f"**Highest Accuracy:** {best_accuracy['Model']} "
            f"({best_accuracy['Accuracy']:.3f})"
        )

        st.info(
            f"**Highest Precision:** {best_precision['Model']} "
            f"({best_precision['Precision']:.3f})"
        )

    with insight_col2:

        st.info(
            f"**Highest Recall:** {best_recall['Model']} "
            f"({best_recall['Recall']:.3f})"
        )

        st.info(
            f"**Highest ROC-AUC:** {best_roc['Model']} "
            f"({best_roc['ROC-AUC']:.3f})"
        )

    st.success(
        "Model evaluation helps identify the balance between "
        "correct predictions, identifying churn customers and "
        "reducing incorrect predictions."
    )

# =========================================================
# 16. ABOUT PROJECT
# =========================================================

elif page == "About Project":

    page_heading(
        "About ChurnIQ",
        "An intelligent customer churn prediction and retention analytics platform."
    )

    # PROJECT INTRODUCTION

    with st.container(border=True):

        st.subheader("Customer Churn Prediction System")

        st.write(
            "ChurnIQ is a machine learning-based application designed "
            "to identify customers who are likely to discontinue a company's services."
        )

        st.write(
            "The platform analyzes customer information, estimates churn "
            "probability, identifies high-risk customer groups and supports "
            "data-driven retention decisions."
        )

    st.markdown("### Key Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:

        with st.container(border=True):
            st.subheader("01 · Customer Analytics")
            st.write(
                "Explore customer distribution, churn patterns, "
                "contract types and customer behavior."
            )

        with st.container(border=True):
            st.subheader("02 · Individual Prediction")
            st.write(
                "Enter customer details and estimate the probability "
                "of customer churn."
            )

        with st.container(border=True):
            st.subheader("03 · Batch Prediction")
            st.write(
                "Upload customer records through CSV and generate "
                "predictions for multiple customers."
            )

    with feature_col2:

        with st.container(border=True):
            st.subheader("04 · Retention Analytics")
            st.write(
                "Identify high-risk customer segments and explore "
                "retention opportunities."
            )

        with st.container(border=True):
            st.subheader("05 · Model Evaluation")
            st.write(
                "Review model evaluation metrics, confusion matrix "
                "and classification performance."
            )

        with st.container(border=True):
            st.subheader("06 · Interactive Dashboard")
            st.write(
                "Explore customer insights through an interactive "
                "and easy-to-use dashboard."
            )

    st.divider()

    # TECHNOLOGY STACK

    st.markdown("### Technology Stack")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:
        with st.container(border=True):
            st.markdown("**Programming**")
            st.write("Python")
            st.write("Pandas")
            st.write("NumPy")

    with tech_col2:
        with st.container(border=True):
            st.markdown("**Machine Learning**")
            st.write("Scikit-learn")
            st.write("Classification Models")
            st.write("Model Evaluation")

    with tech_col3:
        with st.container(border=True):
            st.markdown("**Visualization & Deployment**")
            st.write("Streamlit")
            st.write("Plotly")
            st.write("Matplotlib")

    st.divider()

    # PROJECT OBJECTIVE

    st.markdown("### Project Objective")

    st.write(
        "The main objective of ChurnIQ is to transform customer data "
        "into actionable insights. By identifying customers with a higher "
        "likelihood of leaving, businesses can prioritize retention "
        "activities and make better customer relationship decisions."
    )

    st.divider()

    # FOOTER

    with st.container(border=True):
        st.title("ChurnIQ")

        st.caption(
            "Customer Churn Prediction & Retention Analytics"
        )

        st.caption(
            "Built with Python, Machine Learning and Streamlit"
        )


# =========================================================
# END OF APPLICATION
# =========================================================
