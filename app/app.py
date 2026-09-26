import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ChurnAI | Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_PATH = BASE_DIR / "outputs" / "results" / "model_comparison.csv"
FEATURE_PATH = BASE_DIR / "outputs" / "results" / "feature_importance.csv"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background: linear-gradient(
            135deg,
            #0b1220 0%,
            #111827 45%,
            #172554 100%
        );
        color: #f8fafc;
    }

    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0f172a 0%,
            #111827 100%
        );
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Hero ---------- */

    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 24px;
        background:
            linear-gradient(
                135deg,
                rgba(37,99,235,0.95),
                rgba(79,70,229,0.90)
            );
        box-shadow: 0 18px 45px rgba(0,0,0,0.25);
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255,255,255,0.12);
    }

    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        color: white;
    }

    .hero p {
        font-size: 1.05rem;
        color: rgba(255,255,255,0.86);
        margin: 0;
    }

    /* ---------- Section Headers ---------- */

    .section-title {
        font-size: 1.5rem;
        font-weight: 750;
        margin-top: 1.3rem;
        margin-bottom: 0.9rem;
        color: #e2e8f0;
    }

    .section-subtitle {
        color: #94a3b8;
        margin-bottom: 1rem;
    }

    /* ---------- KPI Cards ---------- */

    .metric-card {
        padding: 1.3rem 1.4rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 28px rgba(0,0,0,0.15);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 800;
    }

    /* ---------- Prediction Cards ---------- */

    .prediction-churn {
        padding: 2rem;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            rgba(127,29,29,0.85),
            rgba(190,24,93,0.78)
        );
        border: 1px solid rgba(251,113,133,0.45);
        text-align: center;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }

    .prediction-stay {
        padding: 2rem;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            rgba(6,78,59,0.88),
            rgba(5,150,105,0.72)
        );
        border: 1px solid rgba(52,211,153,0.4);
        text-align: center;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }

    .prediction-title {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.4rem;
    }

    .prediction-prob {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
    }

    /* ---------- Info Cards ---------- */

    .info-card {
        padding: 1.3rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        min-height: 145px;
    }

    .info-card h4 {
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }

    .info-card p {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* ---------- Risk Pills ---------- */

    .risk-high {
        display: inline-block;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        background: #7f1d1d;
        color: #fecaca;
        font-weight: 700;
    }

    .risk-medium {
        display: inline-block;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        background: #78350f;
        color: #fde68a;
        font-weight: 700;
    }

    .risk-low {
        display: inline-block;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        background: #064e3b;
        color: #a7f3d0;
        font-weight: 700;
    }

    /* ---------- Sidebar ---------- */

    .sidebar-logo {
        text-align: center;
        padding: 1rem 0 1.5rem 0;
    }

    .sidebar-logo h2 {
        margin: 0;
        color: white;
        font-weight: 800;
    }

    .sidebar-logo p {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    /* ---------- Buttons ---------- */

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 0.75rem;
        font-weight: 750;
        font-size: 1rem;
    }

    /* ---------- Tables ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 15px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)


@st.cache_data
def load_feature_importance():
    return pd.read_csv(FEATURE_PATH)


@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(
            MODEL_DIR / "logistic_regression.pkl"
        ),
        "Decision Tree": joblib.load(
            MODEL_DIR / "decision_tree.pkl"
        ),
        "Random Forest": joblib.load(
            MODEL_DIR / "random_forest.pkl"
        )
    }


df = load_data()
results_df = load_results()
saved_feature_importance = load_feature_importance()
models = load_models()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_model_features(model_pipeline):
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_

    elif hasattr(model, "coef_"):
        importance_values = np.abs(model.coef_[0])

    else:
        return pd.DataFrame(
            columns=["Feature", "Importance"]
        )

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    feature_df["Feature"] = (
        feature_df["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    return feature_df


def get_risk_level(probability):
    if probability >= 0.70:
        return "High Risk", "risk-high"

    if probability >= 0.40:
        return "Medium Risk", "risk-medium"

    return "Low Risk", "risk-low"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-logo">
            <h2>📊 ChurnAI</h2>
            <p>Customer Churn Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔮 Customer Prediction",
            "📈 Model Performance",
            "🔎 Churn Analytics"
        ]
    )

    st.markdown("---")

    st.markdown(
        """
        **Machine Learning Models**

        • Logistic Regression  
        • Decision Tree  
        • Random Forest
        """
    )

    st.markdown("---")

    st.caption("Telecom Customer Churn Project")


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        """
        <div class="hero">
            <h1>Customer Churn Prediction</h1>
            <p>
            An intelligent machine-learning dashboard for analysing
            telecom customer behaviour and predicting potential churn.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_customers = len(df)
    churned = int((df["Churn"] == "Yes").sum())
    stayed = int((df["Churn"] == "No").sum())
    churn_rate = churned / total_customers * 100
    avg_monthly = df["MonthlyCharges"].mean()

    # KPI ROW

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Customers</div>
                <div class="metric-value">{total_customers:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Churned Customers</div>
                <div class="metric-value">{churned:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Overall Churn Rate</div>
                <div class="metric-value">{churn_rate:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Monthly Charges</div>
                <div class="metric-value">${avg_monthly:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">Customer Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        pie_data = pd.DataFrame({
            "Status": ["No Churn", "Churn"],
            "Customers": [stayed, churned]
        })

        fig = px.pie(
            pie_data,
            names="Status",
            values="Customers",
            hole=0.58,
            title="Customer Churn Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        contract = pd.crosstab(
            df["Contract"],
            df["Churn"],
            normalize="index"
        ) * 100

        contract = contract.reset_index()

        fig = px.bar(
            contract,
            x="Contract",
            y="Yes",
            text_auto=".1f",
            title="Churn Rate by Contract"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Churn Rate (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown(
        '<div class="section-title">Project Highlights</div>',
        unsafe_allow_html=True
    )

    i1, i2, i3 = st.columns(3)

    with i1:
        st.markdown(
            """
            <div class="info-card">
                <h4>📊 Data Analytics</h4>
                <p>
                Analyse customer demographics, services, contracts,
                tenure and billing behaviour.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i2:
        st.markdown(
            """
            <div class="info-card">
                <h4>🤖 Machine Learning</h4>
                <p>
                Compare Logistic Regression, Decision Tree and
                Random Forest models.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i3:
        st.markdown(
            """
            <div class="info-card">
                <h4>🔮 Prediction</h4>
                <p>
                Enter customer details and estimate the probability
                of customer churn.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

elif page == "🔮 Customer Prediction":

    st.markdown(
        """
        <div class="hero">
            <h1>🔮 Customer Churn Prediction</h1>
            <p>
            Enter customer details to generate an individual churn
            prediction using a trained machine-learning pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    model_name = st.selectbox(
        "Select prediction model",
        [
            "Random Forest",
            "Logistic Regression",
            "Decision Tree"
        ]
    )

    selected_model = models[model_name]

    # CUSTOMER INFORMATION

    st.markdown(
        '<div class="section-title">👤 Customer Information</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

    with c2:
        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with c3:
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )

    c1, c2 = st.columns(2)

    with c1:
        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

    with c2:
        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

    # SERVICE INFORMATION

    st.markdown(
        '<div class="section-title">🌐 Service Information</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    with c2:

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with c3:

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    # ACCOUNT INFORMATION

    st.markdown(
        '<div class="section-title">💳 Account Information</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

    with c2:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with c3:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    c1, c2 = st.columns(2)

    with c1:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=18.25,
            max_value=118.75,
            value=70.00,
            step=0.01
        )

    with c2:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=round(
                monthly_charges * tenure,
                2
            ),
            step=0.01
        )

    st.markdown("")

    predict = st.button(
        "🚀 ANALYZE CUSTOMER CHURN",
        use_container_width=True
    )

    if predict:

        input_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]
        })

        prediction = int(
            selected_model.predict(input_data)[0]
        )

        probability = float(
            selected_model.predict_proba(input_data)[0][1]
        )

        risk_level, risk_class = get_risk_level(probability)

        st.markdown("---")

        if prediction == 1:

            st.markdown(
                f"""
                <div class="prediction-churn">
                    <div class="prediction-title">
                        ⚠️ CUSTOMER LIKELY TO CHURN
                    </div>
                    <div class="prediction-probability">
                        Predicted churn probability:
                        <strong>{probability * 100:.2f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="prediction-stay">
                    <div class="prediction-title">
                        ✅ CUSTOMER LIKELY TO STAY
                    </div>
                    <div class="prediction-probability">
                        Predicted churn probability:
                        <strong>{probability * 100:.2f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Selected Model</div>
                    <div class="metric-value" style="font-size:1.25rem;">
                        {model_name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Level</div>
                    <div style="margin-top:0.5rem;">
                        <span class="{risk_class}">
                            {risk_level}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Customer Tenure</div>
                    <div class="metric-value">
                        {tenure} months
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # GAUGE

        st.markdown(
            '<div class="section-title">🎯 Churn Probability</div>',
            unsafe_allow_html=True
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={
                    "suffix": "%",
                    "font": {
                        "size": 42
                    }
                },
                title={
                    "text": "Probability of Churn"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "thickness": 0.28
                    },
                    "steps": [
                        {
                            "range": [0, 40]
                        },
                        {
                            "range": [40, 70]
                        },
                        {
                            "range": [70, 100]
                        }
                    ],
                    "threshold": {
                        "line": {
                            "width": 5
                        },
                        "value": 50
                    }
                }
            )
        )

        gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=340
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        # CUSTOMER SUMMARY

        st.markdown(
            '<div class="section-title">📋 Customer Summary</div>',
            unsafe_allow_html=True
        )

        summary = pd.DataFrame({
            "Attribute": [
                "Contract",
                "Tenure",
                "Internet Service",
                "Tech Support",
                "Payment Method",
                "Monthly Charges"
            ],
            "Value": [
                contract,
                f"{tenure} months",
                internet_service,
                tech_support,
                payment_method,
                f"${monthly_charges:.2f}"
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        # MODEL FEATURES

        st.markdown(
            '<div class="section-title">🔎 Important Model Features</div>',
            unsafe_allow_html=True
        )

        model_feature_df = get_model_features(
            selected_model
        ).head(10)

        fig = px.bar(
            model_feature_df.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top Features — {model_name}"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.markdown(
        """
        <div class="hero">
            <h1>📈 Model Performance</h1>
            <p>
            Compare the classification models using multiple
            performance metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    display_df = results_df.copy()

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]

    for metric in metrics:
        display_df[metric] = (
            display_df[metric] * 100
        ).round(2)

    st.markdown(
        '<div class="section-title">Model Metrics</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">📊 Performance Comparison</div>',
        unsafe_allow_html=True
    )

    long_df = results_df.melt(
        id_vars="Model",
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )

    long_df["Score"] *= 100

    fig = px.bar(
        long_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text_auto=".1f",
        title="Model Performance Across Evaluation Metrics"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_range=[0, 100]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">🎯 Confusion Matrices</div>',
        unsafe_allow_html=True
    )

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    X = df.drop(
        columns=["customerID", "Churn"]
    )

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    for model_name, model in models.items():

        predictions = model.predict(X_test)

        cm = confusion_matrix(
            y_test,
            predictions
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual No Churn",
                "Actual Churn"
            ],
            columns=[
                "Predicted No Churn",
                "Predicted Churn"
            ]
        )

        st.markdown(
            f"#### {model_name}"
        )

        fig = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            title=f"Confusion Matrix — {model_name}"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# CHURN ANALYTICS
# ============================================================

elif page == "🔎 Churn Analytics":

    st.markdown(
        """
        <div class="hero">
            <h1>🔎 Churn Analytics</h1>
            <p>
            Explore the customer characteristics and service
            patterns associated with telecom churn.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    analysis = st.selectbox(
        "Choose an analysis",
        [
            "Churn by Contract",
            "Churn by Internet Service",
            "Churn by Payment Method",
            "Churn by Tech Support",
            "Churn by Online Security",
            "Churn by Paperless Billing",
            "Churn by Senior Citizen",
            "Tenure vs Churn",
            "Monthly Charges vs Churn"
        ]
    )

    if analysis == "Churn by Contract":

        temp = pd.crosstab(
            df["Contract"],
            df["Churn"],
            normalize="index"
        ) * 100

        temp = temp.reset_index()

        fig = px.bar(
            temp,
            x="Contract",
            y="Yes",
            text_auto=".1f",
            title="Churn Rate by Contract Type"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif analysis == "Churn by Internet Service":

        temp = pd.crosstab(
            df["InternetService"],
            df["Churn"],
            normalize="index"
        ) * 100

        temp = temp.reset_index()

        fig = px.bar(
            temp,
            x="InternetService",
            y="Yes",
            text_auto=".1f",
            title="Churn Rate by Internet Service"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif analysis == "Churn by Payment Method":

        temp = pd.crosstab(
            df["PaymentMethod"],
            df["Churn"],
            normalize="index"
        ) * 100

        temp = temp.reset_index()

        fig = px.bar(
            temp,
            x="PaymentMethod",
            y="Yes",
            text_auto=".1f",
            title="Churn Rate by Payment Method"
        )

        fig.update_xaxes(tickangle=25)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif analysis == "Churn by Tech Support":

        temp = pd.crosstab(
            df["TechSupport"],
            df["Churn"],
            normalize="index"
        ) * 100

        temp = temp.reset_index()

        fig = px.bar(
            temp,
            x="TechSupport",
            y="Yes",
            text_auto=".1f",
            title="Churn Rate by Tech Support"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif analysis == "Churn by Online Security":

        temp = pd.crosstab(
            df["OnlineSecurity"],
            df["Churn"],
            normalize="index"
        ) * 100

        temp = temp.reset_index()

        fig = px.bar(
            temp,
            x="OnlineSecurity",
            y="Yes",
            text_auto=".1f",
            title="Churn Rate by Online Security"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif analysis == "Churn by Paperless Billing":

        temp = pd.crosstab(
            df["PaperlessBilling"],
            df["Churn"],
            normalize="index"
        ) * 100

        temp = temp.reset_index()

        fig = px.bar(
            temp,
            x="PaperlessBilling",
            y="Yes",
            text_auto=".1f",
            title="Churn Rate by Paperless Billing"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif analysis == "Churn by Senior Citizen":

        temp = pd.crosstab(
            df["SeniorCitizen"],
            df["Churn"],
            normalize="index"
        ) * 100

        temp = temp.reset_index()

        temp["SeniorCitizen"] = temp["SeniorCitizen"].map({
            0: "No",
            1: "Yes"
        })

        fig = px.bar(
            temp,
            x="SeniorCitizen",
            y="Yes",
            text_auto=".1f",
            title="Churn Rate by Senior Citizen Status"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif analysis == "Tenure vs Churn":

        fig = px.box(
            df,
            x="Churn",
            y="tenure",
            color="Churn",
            title="Tenure Distribution by Churn Status"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif analysis == "Monthly Charges vs Churn":

        fig = px.box(
            df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            title="Monthly Charges by Churn Status"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        padding:1rem;
        font-size:0.85rem;
    ">
        <b>ChurnAI</b> — Telecom Customer Churn Prediction & Analytics
        <br>
        Machine Learning Academic Project
    </div>
    """,
    unsafe_allow_html=True
)