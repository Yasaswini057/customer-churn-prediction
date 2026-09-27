import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ChurnInsight | Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "telco_churn_cleaned.csv"
)

MODEL_DIR = BASE_DIR / "models"

RESULTS_PATH = (
    BASE_DIR
    / "outputs"
    / "results"
    / "model_comparison.csv"
)

FEATURE_PATH = (
    BASE_DIR
    / "outputs"
    / "results"
    / "feature_importance.csv"
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Light"


# ============================================================
# CURRENT THEME
# ============================================================

theme = st.session_state.ui_theme


if theme == "Dark":

    APP_BG = "#0B1120"
    CARD_BG = "#111827"
    BORDER = "#263449"

    TEXT_MAIN = "#F8FAFC"
    TEXT_SECONDARY = "#94A3B8"

    INPUT_BG = "#172033"

    PRIMARY = "#818CF8"
    PRIMARY_DARK = "#4F46E5"

    DANGER = "#FB7185"

    CHART_TEMPLATE = "plotly_dark"

else:

    APP_BG = "#F5F7FB"
    CARD_BG = "#FFFFFF"
    BORDER = "#E2E8F0"

    TEXT_MAIN = "#172033"
    TEXT_SECONDARY = "#64748B"

    INPUT_BG = "#FFFFFF"

    PRIMARY = "#4F46E5"
    PRIMARY_DARK = "#3730A3"

    DANGER = "#E11D48"

    CHART_TEMPLATE = "plotly_white"


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background: {APP_BG};
    }}

    .block-container {{
        max-width: 1550px;
        width: 100%;
        padding-top: 1rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        padding-bottom: 2rem;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {{
        width: 270px !important;
        background: {CARD_BG};
        border-right: 1px solid {BORDER};
    }}

    section[data-testid="stSidebar"] > div:first-child {{
        padding-top: 0.35rem;
    }}

    section[data-testid="stSidebar"] * {{
        color: {TEXT_MAIN};
    }}


    /* ======================================================
       BRAND HIGHLIGHT
       ====================================================== */

    .brand-box {{
        background: linear-gradient(
            135deg,
            {PRIMARY_DARK},
            {PRIMARY}
        );

        border-radius: 15px;

        padding: 0.85rem 0.9rem;

        margin-bottom: 1rem;

        box-shadow:
            0 8px 22px rgba(79,70,229,0.20);
    }}

    .brand-name {{
        color: white !important;

        font-size: 1.55rem;

        font-weight: 850;

        letter-spacing: -0.4px;

        line-height: 1;

        margin: 0;
    }}

    .brand-subtitle {{
        color: rgba(255,255,255,0.82) !important;

        font-size: 0.72rem;

        margin-top: 0.3rem;

        margin-bottom: 0;
    }}


    /* ======================================================
       SIDEBAR NAVIGATION
       ====================================================== */

    section[data-testid="stSidebar"]
    .stButton > button {{

        width: 100%;

        min-height: 43px;

        border-radius: 11px;

        border: 1px solid {BORDER};

        background: {CARD_BG};

        color: {TEXT_MAIN};

        font-weight: 650;

        text-align: left;

        margin-bottom: 0.35rem;

        transition:
            background 0.18s ease,
            border-color 0.18s ease,
            transform 0.18s ease;
    }}

    section[data-testid="stSidebar"]
    .stButton > button:hover {{

        border-color: {PRIMARY};

        background: rgba(79,70,229,0.08);

        transform: translateX(2px);
    }}


    /* ======================================================
       ACTIVE NAVIGATION
       ====================================================== */

    section[data-testid="stSidebar"]
    button[kind="primary"] {{

        background: {PRIMARY_DARK};

        color: white;

        border-color: {PRIMARY_DARK};

        box-shadow:
            0 7px 18px rgba(79,70,229,0.20);
    }}


    /* ======================================================
       MAIN HEADINGS
       ====================================================== */

    h1 {{
        color: {TEXT_MAIN} !important;

        font-weight: 820 !important;

        letter-spacing: -0.7px;
    }}

    h2 {{
        color: {TEXT_MAIN} !important;

        font-weight: 780 !important;
    }}

    h3 {{
        color: {TEXT_MAIN} !important;

        font-weight: 740 !important;
    }}

    p {{
        color: {TEXT_MAIN};
    }}


    /* ======================================================
       MAIN TOP TITLE
       ====================================================== */

    .top-caption {{
        color: {TEXT_SECONDARY};

        font-size: 0.88rem;

        margin-top: -0.4rem;
    }}


    /* ======================================================
       HIGHLIGHTED HEADINGS
       ====================================================== */

    .heading-box {{
        background: linear-gradient(
            135deg,
            rgba(79,70,229,0.12),
            rgba(20,184,166,0.10)
        );

        border-left: 5px solid {PRIMARY};

        border-radius: 14px;

        padding: 1.05rem 1.25rem;

        margin: 1rem 0 1.35rem 0;
    }}

    .heading-box h2 {{
        margin: 0;

        color: {TEXT_MAIN};

        font-size: 1.7rem;

        font-weight: 800;
    }}

    .heading-box p {{
        margin: 0.35rem 0 0 0;

        color: {TEXT_SECONDARY};

        font-size: 0.92rem;
    }}


    /* ======================================================
       CONTAINERS
       ====================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {CARD_BG};

        border: 1px solid {BORDER};

        border-radius: 16px;
    }}


    /* ======================================================
       METRICS
       ====================================================== */

    div[data-testid="stMetric"] {{

        background: {CARD_BG};

        border: 1px solid {BORDER};

        border-radius: 16px;

        padding: 1rem 1.05rem;

        box-shadow:
            0 7px 22px rgba(15,23,42,0.045);
    }}

    div[data-testid="stMetricLabel"] {{
        color: {TEXT_SECONDARY} !important;
    }}

    div[data-testid="stMetricValue"] {{
        color: {TEXT_MAIN} !important;

        font-weight: 820 !important;
    }}


    /* ======================================================
       INPUTS
       ====================================================== */

    .stSelectbox label,
    .stNumberInput label {{

        color: {TEXT_MAIN} !important;

        font-weight: 650 !important;
    }}

    div[data-baseweb="select"] > div {{
        background: {INPUT_BG};

        border-radius: 10px;
    }}

    input {{
        background: {INPUT_BG} !important;

        color: {TEXT_MAIN} !important;

        border-radius: 10px !important;
    }}


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {{

        border-radius: 11px;

        min-height: 44px;

        font-weight: 700;

        transition: 0.18s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
    }}


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer-text {{

        text-align: center;

        color: {TEXT_SECONDARY};

        font-size: 0.78rem;

        padding-top: 1rem;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():

        st.error(
            f"Dataset not found:\n{DATA_PATH}"
        )

        st.stop()

    return pd.read_csv(DATA_PATH)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    paths = {

        "Logistic Regression":
            MODEL_DIR / "logistic_regression.pkl",

        "Decision Tree":
            MODEL_DIR / "decision_tree.pkl",

        "Random Forest":
            MODEL_DIR / "random_forest.pkl"
    }


    missing = [

        str(path)

        for path in paths.values()

        if not path.exists()
    ]


    if missing:

        st.error(
            "Missing trained model files:\n\n"
            + "\n".join(missing)
        )

        st.stop()


    return {

        name: joblib.load(path)

        for name, path in paths.items()
    }


# ============================================================
# LOAD RESULTS
# ============================================================

@st.cache_data
def load_results():

    if not RESULTS_PATH.exists():

        st.error(
            f"Model comparison file not found:\n{RESULTS_PATH}"
        )

        st.stop()

    return pd.read_csv(
        RESULTS_PATH
    )


# ============================================================
# LOAD FEATURE IMPORTANCE
# ============================================================

@st.cache_data
def load_feature_importance():

    if not FEATURE_PATH.exists():

        st.error(
            f"Feature importance file not found:\n{FEATURE_PATH}"
        )

        st.stop()

    return pd.read_csv(
        FEATURE_PATH
    )


# ============================================================
# INITIALIZE
# ============================================================

df = load_data()

models = load_models()

results_df = load_results()

saved_feature_importance = (
    load_feature_importance()
)


# ============================================================
# EVALUATION DATA
# ============================================================

X_eval = df.drop(
    columns=[
        "customerID",
        "Churn"
    ]
)

y_eval = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


X_train_eval, X_test_eval, y_train_eval, y_test_eval = (
    train_test_split(

        X_eval,

        y_eval,

        test_size=0.20,

        random_state=42,

        stratify=y_eval
    )
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_model_features(model_pipeline):

    preprocessor = (
        model_pipeline
        .named_steps["preprocessor"]
    )

    model = (
        model_pipeline
        .named_steps["model"]
    )


    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    if hasattr(
        model,
        "feature_importances_"
    ):

        values = (
            model.feature_importances_
        )


    elif hasattr(
        model,
        "coef_"
    ):

        values = np.abs(
            model.coef_[0]
        )


    else:

        return pd.DataFrame(
            columns=[
                "Feature",
                "Importance"
            ]
        )


    feature_df = pd.DataFrame({

        "Feature":
            feature_names,

        "Importance":
            values
    })


    feature_df = (
        feature_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    feature_df["Feature"] = (

        feature_df["Feature"]

        .str.replace(
            "num__",
            "",
            regex=False
        )

        .str.replace(
            "cat__",
            "",
            regex=False
        )
    )


    return feature_df


def get_risk_level(
    probability
):

    if probability >= 0.70:

        return "High Risk"

    elif probability >= 0.40:

        return "Medium Risk"

    return "Low Risk"


def make_rate_table(
    column
):

    table = pd.crosstab(

        df[column],

        df["Churn"],

        normalize="index"
    ) * 100


    return table.reset_index()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # BRAND AT VERY TOP

    st.markdown(
        '<div class="brand-box">'
        '<div class="brand-name">ChurnInsight</div>'
        '<div class="brand-subtitle">'
        'Customer Churn Intelligence'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "NAVIGATION"
    )


    navigation = {

        "Dashboard":
            "🏠  Dashboard",

        "Prediction":
            "🔮  Customer Prediction",

        "Performance":
            "📈  Model Performance",

        "Analytics":
            "🔎  Churn Analytics"
    }


    for key, label in navigation.items():

        if st.button(

            label,

            key=f"nav_{key}",

            width="stretch",

            type=(

                "primary"

                if st.session_state.page
                == key

                else "secondary"
            )
        ):

            st.session_state.page = key

            st.rerun()


# ============================================================
# TOP HEADER
# ============================================================

header_left, header_right = (
    st.columns(
        [7, 1]
    )
)


with header_left:

    st.title(
        "Customer Churn Prediction"
    )

    st.markdown(
        '<div class="top-caption">'
        'Telecom Analytics • Machine Learning • '
        'Customer Churn Prediction'
        '</div>',
        unsafe_allow_html=True
    )


with header_right:

    # APPEARANCE

    with st.popover(
        "Appearance",
        width="content"
    ):

        st.subheader(
            "Appearance"
        )

        st.caption(
            "Change the application theme."
        )


        light_col, dark_col = (
            st.columns(2)
        )


        with light_col:

            if st.button(
                "☀ Light",
                width="stretch",
                key="theme_light_button"
            ):

                st.session_state.ui_theme = (
                    "Light"
                )

                st.rerun()


        with dark_col:

            if st.button(
                "🌙 Dark",
                width="stretch",
                key="theme_dark_button"
            ):

                st.session_state.ui_theme = (
                    "Dark"
                )

                st.rerun()


        st.caption(
            f"Current theme: {st.session_state.ui_theme}"
        )


st.divider()


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    # HIGHLIGHTED MAIN HEADING

    st.markdown(
        """
        <div class="heading-box">

        <h2>
        Customer Churn Intelligence
        </h2>

        <p>
        Explore customer behaviour, understand churn patterns,
        compare classification models and estimate individual
        customer churn probability.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    total_customers = len(df)


    churned_customers = int(
        (
            df["Churn"]
            == "Yes"
        ).sum()
    )


    retained_customers = int(
        (
            df["Churn"]
            == "No"
        ).sum()
    )


    churn_rate = (

        churned_customers
        / total_customers
        * 100
    )


    avg_monthly = float(
        df["MonthlyCharges"].mean()
    )


    avg_tenure_churn = float(
        df.loc[
            df["Churn"] == "Yes",
            "tenure"
        ].mean()
    )


    avg_tenure_retained = float(
        df.loc[
            df["Churn"] == "No",
            "tenure"
        ].mean()
    )


    st.subheader(
        "Portfolio Overview"
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )


    with c2:

        st.metric(
            "Churned Customers",
            f"{churned_customers:,}"
        )


    with c3:

        st.metric(
            "Overall Churn Rate",
            f"{churn_rate:.2f}%"
        )


    with c4:

        st.metric(
            "Average Monthly Charges",
            f"{avg_monthly:.2f}"
        )


    st.subheader(
        "Customer Behaviour"
    )


    chart1, chart2 = (
        st.columns(2)
    )


    with chart1:

        pie_data = pd.DataFrame({

            "Status": [
                "No Churn",
                "Churn"
            ],

            "Customers": [
                retained_customers,
                churned_customers
            ]
        })


        fig = px.pie(

            pie_data,

            names="Status",

            values="Customers",

            hole=0.58,

            title="Churn Distribution",

            color="Status",

            color_discrete_map={

                "No Churn":
                    "#4F46E5",

                "Churn":
                    "#E11D48"
            }
        )


        fig.update_layout(

            template=CHART_TEMPLATE,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    with chart2:

        contract_rate = make_rate_table(
            "Contract"
        )


        fig = px.bar(

            contract_rate,

            x="Contract",

            y="Yes",

            text_auto=".1f",

            title="Churn Rate by Contract",

            color="Contract",

            color_discrete_sequence=[

                "#4F46E5",

                "#14B8A6",

                "#F59E0B"
            ]
        )


        fig.update_layout(

            template=CHART_TEMPLATE,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            showlegend=False,

            xaxis_title="",

            yaxis_title="Churn Rate (%)"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    st.subheader(
        "Key Analytical Insights"
    )


    highest_contract = (
        contract_rate
        .sort_values(
            "Yes",
            ascending=False
        )
        .iloc[0]
    )


    c1, c2, c3 = (
        st.columns(3)
    )


    with c1:

        with st.container(
            border=True
        ):

            st.metric(
                "Highest Contract Churn",
                f"{highest_contract['Yes']:.1f}%"
            )

            st.caption(
                str(
                    highest_contract[
                        "Contract"
                    ]
                )
            )


    with c2:

        with st.container(
            border=True
        ):

            st.metric(
                "Avg. Tenure — Churned",
                f"{avg_tenure_churn:.1f} months"
            )


    with c3:

        with st.container(
            border=True
        ):

            st.metric(
                "Avg. Tenure — Retained",
                f"{avg_tenure_retained:.1f} months"
            )


# ============================================================
# CUSTOMER PREDICTION
# ============================================================

elif st.session_state.page == "Prediction":

    with st.container(
        border=True
    ):

        st.caption(
            "INDIVIDUAL CUSTOMER ANALYSIS"
        )

        st.header(
            "Customer Churn Prediction"
        )

        st.write(
            "Enter customer information and estimate "
            "the likelihood of churn using a trained "
            "classification model."
        )


    model_name = st.selectbox(

        "Prediction Model",

        [
            "Random Forest",
            "Logistic Regression",
            "Decision Tree"
        ]
    )


    selected_model = models[
        model_name
    ]


    # CUSTOMER INFORMATION

    with st.container(
        border=True
    ):

        st.subheader(
            "👤 Customer Information"
        )


        c1, c2, c3 = (
            st.columns(3)
        )


        with c1:

            gender = st.selectbox(
                "Gender",
                [
                    "Female",
                    "Male"
                ]
            )


        with c2:

            senior_citizen = st.selectbox(

                "Senior Citizen",

                [
                    0,
                    1
                ],

                format_func=lambda x:

                "Yes"
                if x == 1
                else "No"
            )


        with c3:

            tenure = st.number_input(

                "Tenure (Months)",

                min_value=0,

                max_value=72,

                value=12
            )


        c1, c2 = (
            st.columns(2)
        )


        with c1:

            partner = st.selectbox(

                "Partner",

                [
                    "Yes",
                    "No"
                ]
            )


        with c2:

            dependents = st.selectbox(

                "Dependents",

                [
                    "Yes",
                    "No"
                ]
            )


    # SERVICE INFORMATION

    with st.container(
        border=True
    ):

        st.subheader(
            "🌐 Service Information"
        )


        c1, c2, c3 = (
            st.columns(3)
        )


        with c1:

            phone_service = st.selectbox(

                "Phone Service",

                [
                    "Yes",
                    "No"
                ]
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

    with st.container(
        border=True
    ):

        st.subheader(
            "💳 Account Information"
        )


        c1, c2, c3 = (
            st.columns(3)
        )


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

                [
                    "Yes",
                    "No"
                ]
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


        c1, c2 = (
            st.columns(2)
        )


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

                value=float(
                    round(
                        monthly_charges
                        * tenure,
                        2
                    )
                ),

                step=0.01
            )


    st.write("")


    predict = st.button(

        "🚀  ANALYZE CUSTOMER CHURN",

        width="stretch",

        type="primary"
    )


    if predict:

        input_data = pd.DataFrame({

            "gender": [
                gender
            ],

            "SeniorCitizen": [
                senior_citizen
            ],

            "Partner": [
                partner
            ],

            "Dependents": [
                dependents
            ],

            "tenure": [
                tenure
            ],

            "PhoneService": [
                phone_service
            ],

            "MultipleLines": [
                multiple_lines
            ],

            "InternetService": [
                internet_service
            ],

            "OnlineSecurity": [
                online_security
            ],

            "OnlineBackup": [
                online_backup
            ],

            "DeviceProtection": [
                device_protection
            ],

            "TechSupport": [
                tech_support
            ],

            "StreamingTV": [
                streaming_tv
            ],

            "StreamingMovies": [
                streaming_movies
            ],

            "Contract": [
                contract
            ],

            "PaperlessBilling": [
                paperless_billing
            ],

            "PaymentMethod": [
                payment_method
            ],

            "MonthlyCharges": [
                monthly_charges
            ],

            "TotalCharges": [
                total_charges
            ]
        })


        prediction = int(

            selected_model
            .predict(
                input_data
            )[0]
        )


        probability = float(

            selected_model
            .predict_proba(
                input_data
            )[0][1]
        )


        risk = get_risk_level(
            probability
        )


        st.divider()


        st.subheader(
            "Prediction Result"
        )


        result_col1, result_col2 = (
            st.columns(
                [2, 1]
            )
        )


        with result_col1:

            if prediction == 1:

                st.error(

                    "⚠️ **CUSTOMER LIKELY TO CHURN**\n\n"

                    f"Predicted churn probability: "
                    f"**{probability * 100:.2f}%**"
                )

            else:

                st.success(

                    "✅ **CUSTOMER LIKELY TO STAY**\n\n"

                    f"Predicted churn probability: "
                    f"**{probability * 100:.2f}%**"
                )


        with result_col2:

            st.metric(
                "Risk Level",
                risk
            )

            st.metric(
                "Selected Model",
                model_name
            )


        c1, c2, c3 = (
            st.columns(3)
        )


        with c1:

            st.metric(

                "Prediction",

                (
                    "Churn"

                    if prediction == 1

                    else "No Churn"
                )
            )


        with c2:

            st.metric(

                "Churn Probability",

                f"{probability * 100:.2f}%"
            )


        with c3:

            st.metric(

                "Customer Tenure",

                f"{tenure} months"
            )


        st.subheader(
            "Probability Analysis"
        )


        gauge = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=probability * 100,

                number={
                    "suffix": "%"
                },

                title={
                    "text":
                    "Predicted Churn Probability"
                },

                gauge={

                    "axis": {
                        "range": [
                            0,
                            100
                        ]
                    },

                    "bar": {
                        "color":
                        PRIMARY
                    },

                    "steps": [

                        {
                            "range": [
                                0,
                                40
                            ],

                            "color":
                            "#DCFCE7"
                        },

                        {
                            "range": [
                                40,
                                70
                            ],

                            "color":
                            "#FEF3C7"
                        },

                        {
                            "range": [
                                70,
                                100
                            ],

                            "color":
                            "#FFE4E6"
                        }
                    ],

                    "threshold": {

                        "line": {
                            "color":
                            DANGER,

                            "width":
                            4
                        },

                        "value":
                        50
                    }
                }
            )
        )


        gauge.update_layout(

            template=CHART_TEMPLATE,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            height=320
        )


        st.plotly_chart(
            gauge,
            width="stretch"
        )


        # PROFILE SIGNALS

        signals = []


        if tenure <= 12:

            signals.append(
                "Short customer tenure"
            )


        if contract == "Month-to-month":

            signals.append(
                "Month-to-month contract"
            )


        if monthly_charges >= 75:

            signals.append(
                "Higher monthly charges"
            )


        if tech_support == "No":

            signals.append(
                "No technical support"
            )


        if online_security == "No":

            signals.append(
                "No online security"
            )


        with st.container(
            border=True
        ):

            st.subheader(
                "🔎 Customer Profile Signals"
            )

            st.caption(
                "These indicators are based on customer attributes "
                "and observed project patterns. They do not establish causation."
            )


            if signals:

                signal_cols = st.columns(
                    min(
                        len(signals),
                        3
                    )
                )


                for index, signal in enumerate(
                    signals[:3]
                ):

                    with signal_cols[index]:

                        st.warning(
                            signal
                        )

            else:

                st.success(
                    "No highlighted profile indicators "
                    "were identified."
                )


        st.subheader(
            "Customer Summary"
        )


        summary = pd.DataFrame({

            "Attribute": [

                "Contract",

                "Tenure",

                "Internet Service",

                "Tech Support",

                "Online Security",

                "Payment Method",

                "Monthly Charges",

                "Total Charges"
            ],

            "Value": [

                contract,

                f"{tenure} months",

                internet_service,

                tech_support,

                online_security,

                payment_method,

                f"{monthly_charges:.2f}",

                f"{total_charges:.2f}"
            ]
        })


        st.dataframe(

            summary,

            width="stretch",

            hide_index=True
        )


        st.subheader(
            "🧠 Important Model Features"
        )


        model_features = (
            get_model_features(
                selected_model
            )
            .head(10)
        )


        if not model_features.empty:

            fig = px.bar(

                model_features
                .sort_values(
                    "Importance"
                ),

                x="Importance",

                y="Feature",

                orientation="h",

                title=(
                    f"Top Features — "
                    f"{model_name}"
                ),

                color="Importance",

                color_continuous_scale=[

                    "#C7D2FE",

                    "#4F46E5"
                ]
            )


            fig.update_layout(

                template=CHART_TEMPLATE,

                paper_bgcolor="rgba(0,0,0,0)",

                plot_bgcolor="rgba(0,0,0,0)",

                yaxis_title="",

                xaxis_title="Importance",

                coloraxis_showscale=False
            )


            st.plotly_chart(
                fig,
                width="stretch"
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif st.session_state.page == "Performance":

    with st.container(
        border=True
    ):

        st.caption(
            "MODEL EVALUATION"
        )

        st.header(
            "Machine Learning Performance"
        )

        st.write(
            "Compare the trained classification models "
            "using Accuracy, Precision, Recall, F1-Score "
            "and ROC-AUC."
        )


    metrics = [

        "Accuracy",

        "Precision",

        "Recall",

        "F1-Score",

        "ROC-AUC"
    ]


    display_df = results_df.copy()


    for metric in metrics:

        display_df[metric] = (

            display_df[metric]
            * 100

        ).round(2)


    st.subheader(
        "Model Comparison"
    )


    st.dataframe(

        display_df,

        width="stretch",

        hide_index=True
    )


    accuracy_model = (
        results_df.loc[
            results_df["Accuracy"].idxmax(),
            "Model"
        ]
    )


    recall_model = (
        results_df.loc[
            results_df["Recall"].idxmax(),
            "Model"
        ]
    )


    f1_model = (
        results_df.loc[
            results_df["F1-Score"].idxmax(),
            "Model"
        ]
    )


    auc_model = (
        results_df.loc[
            results_df["ROC-AUC"].idxmax(),
            "Model"
        ]
    )


    c1, c2, c3, c4 = (
        st.columns(4)
    )


    with c1:

        st.metric(
            "Highest Accuracy",
            accuracy_model
        )


    with c2:

        st.metric(
            "Highest Recall",
            recall_model
        )


    with c3:

        st.metric(
            "Highest F1-Score",
            f1_model
        )


    with c4:

        st.metric(
            "Highest ROC-AUC",
            auc_model
        )


    st.subheader(
        "📊 Performance Comparison"
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

        title="Model Performance Across Metrics",

        color_discrete_sequence=[

            "#4F46E5",

            "#14B8A6",

            "#F59E0B"
        ]
    )


    fig.update_layout(

        template=CHART_TEMPLATE,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        yaxis_title="Score (%)",

        xaxis_title="",

        yaxis_range=[
            0,
            100
        ]
    )


    st.plotly_chart(
        fig,
        width="stretch"
    )


    st.subheader(
        "📈 ROC Curve Comparison"
    )


    roc_fig = go.Figure()


    for model_name, model in models.items():

        probabilities = (

            model.predict_proba(
                X_test_eval
            )[:, 1]
        )


        fpr, tpr, _ = roc_curve(

            y_test_eval,

            probabilities
        )


        roc_auc_value = auc(

            fpr,

            tpr
        )


        roc_fig.add_trace(

            go.Scatter(

                x=fpr,

                y=tpr,

                mode="lines",

                name=(

                    f"{model_name} "
                    f"(AUC = {roc_auc_value:.3f})"
                )
            )
        )


    roc_fig.add_trace(

        go.Scatter(

            x=[
                0,
                1
            ],

            y=[
                0,
                1
            ],

            mode="lines",

            name="Random baseline",

            line={
                "dash":
                "dash"
            }
        )
    )


    roc_fig.update_layout(

        template=CHART_TEMPLATE,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="rgba(0,0,0,0)",

        title="Receiver Operating Characteristic",

        xaxis_title="False Positive Rate",

        yaxis_title="True Positive Rate"
    )


    st.plotly_chart(
        roc_fig,
        width="stretch"
    )


    st.subheader(
        "🎯 Confusion Matrices"
    )


    matrix_cols = st.columns(3)


    for column, (
        model_name,
        model
    ) in zip(
        matrix_cols,
        models.items()
    ):


        with column:

            y_pred = model.predict(
                X_test_eval
            )


            cm = confusion_matrix(

                y_test_eval,

                y_pred
            )


            fig = px.imshow(

                cm,

                text_auto=True,

                aspect="auto",

                title=model_name,

                x=[
                    "No Churn",
                    "Churn"
                ],

                y=[
                    "No Churn",
                    "Churn"
                ],

                color_continuous_scale=[

                    "#EEF2FF",

                    "#4F46E5"
                ]
            )


            fig.update_layout(

                template=CHART_TEMPLATE,

                paper_bgcolor="rgba(0,0,0,0)"
            )


            st.plotly_chart(
                fig,
                width="stretch"
            )


# ============================================================
# CHURN ANALYTICS
# ============================================================

elif st.session_state.page == "Analytics":

    with st.container(
        border=True
    ):

        st.caption(
            "EXPLORATORY DATA ANALYSIS"
        )

        st.header(
            "Churn Analytics"
        )

        st.write(
            "Explore churn patterns across contracts, "
            "services, billing attributes and customer profiles."
        )


    analysis = st.selectbox(

        "Select Analysis",

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

        temp = make_rate_table(
            "Contract"
        )


        fig = px.bar(

            temp,

            x="Contract",

            y="Yes",

            text_auto=".1f",

            title="Churn Rate by Contract Type",

            color="Contract",

            color_discrete_sequence=[

                "#4F46E5",

                "#14B8A6",

                "#F59E0B"
            ]
        )


        fig.update_layout(

            template=CHART_TEMPLATE,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            showlegend=False,

            xaxis_title="",

            yaxis_title="Churn Rate (%)"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    elif analysis == "Churn by Internet Service":

        temp = make_rate_table(
            "InternetService"
        )


        fig = px.bar(

            temp,

            x="InternetService",

            y="Yes",

            text_auto=".1f",

            title="Churn Rate by Internet Service",

            color="InternetService",

            color_discrete_sequence=[

                "#4F46E5",

                "#14B8A6",

                "#F59E0B"
            ]
        )


        fig.update_layout(

            template=CHART_TEMPLATE,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            showlegend=False,

            xaxis_title="",

            yaxis_title="Churn Rate (%)"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    elif analysis == "Churn by Payment Method":

        temp = make_rate_table(
            "PaymentMethod"
        )


        fig = px.bar(

            temp,

            x="PaymentMethod",

            y="Yes",

            text_auto=".1f",

            title="Churn Rate by Payment Method",

            color="PaymentMethod",

            color_discrete_sequence=[

                "#4F46E5",

                "#6366F1",

                "#14B8A6",

                "#F59E0B"
            ]
        )


        fig.update_layout(

            template=CHART_TEMPLATE,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            showlegend=False,

            xaxis_title="",

            yaxis_title="Churn Rate (%)",

            xaxis_tickangle=25
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    elif analysis == "Churn by Tech Support":

        temp = make_rate_table(
            "TechSupport"
        )


        fig = px.bar(

            temp,

            x="TechSupport",

            y="Yes",

            text_auto=".1f",

            title="Churn Rate by Tech Support",

            color="TechSupport",

            color_discrete_sequence=[

                "#4F46E5",

                "#14B8A6"
            ]
        )


        fig.update_layout(

            template=CHART_TEMPLATE,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            showlegend=False,

            xaxis_title="",

            yaxis_title="Churn Rate (%)"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    elif analysis == "Churn by Online Security":

        temp = make_rate_table(
            "OnlineSecurity"
        )


        fig = px.bar(

            temp,

            x="OnlineSecurity",

            y="Yes",

            text_auto=".1f",

            title="Churn Rate by Online Security",

            color="OnlineSecurity",

            color_discrete_sequence=[

                "#4F46E5",

                "#14B8A6"
            ]
        )


        fig.update_layout(

            template=CHART_TEMPLATE,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            showlegend=False,

            xaxis_title="",

            yaxis_title="Churn Rate (%)"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    elif analysis == "Churn by Paperless Billing":

        temp = make_rate_table(
            "PaperlessBilling"
        )


        fig = px.bar(

            temp,

            x="PaperlessBilling",

            y="Yes",

            text_auto=".1f",

            title="Churn Rate by Paperless Billing",

            color="PaperlessBilling",

            color_discrete_sequence=[

                "#4F46E5",

                "#14B8A6"
            ]
        )


        fig.update_layout(

            template=CHART_TEMPLATE,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            showlegend=False,

            xaxis_title="",

            yaxis_title="Churn Rate (%)"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    elif analysis == "Churn by Senior Citizen":

        temp = make_rate_table(
            "SeniorCitizen"
        )


        temp["SeniorCitizen"] = (

            temp["SeniorCitizen"]

            .map({
                0: "No",
                1: "Yes"
            })
        )


        fig = px.bar(

            temp,

            x="SeniorCitizen",

            y="Yes",

            text_auto=".1f",

            title="Churn Rate by Senior Citizen",

            color="SeniorCitizen",

            color_discrete_sequence=[

                "#4F46E5",

                "#14B8A6"
            ]
        )


        fig.update_layout(

            template=CHART_TEMPLATE,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            showlegend=False,

            xaxis_title="",

            yaxis_title="Churn Rate (%)"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    elif analysis == "Tenure vs Churn":

        fig = px.box(

            df,

            x="Churn",

            y="tenure",

            color="Churn",

            title="Tenure Distribution by Churn",

            color_discrete_map={

                "No":
                    "#4F46E5",

                "Yes":
                    "#E11D48"
            }
        )


        fig.update_layout(

            template=CHART_TEMPLATE,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            xaxis_title="Churn",

            yaxis_title="Tenure (Months)"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    elif analysis == "Monthly Charges vs Churn":

        fig = px.box(

            df,

            x="Churn",

            y="MonthlyCharges",

            color="Churn",

            title="Monthly Charges by Churn",

            color_discrete_map={

                "No":
                    "#4F46E5",

                "Yes":
                    "#E11D48"
            }
        )


        fig.update_layout(

            template=CHART_TEMPLATE,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            xaxis_title="Churn",

            yaxis_title="Monthly Charges"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer-text">'
    'ChurnInsight • Customer Churn Prediction & Analytics'
    '</div>',
    unsafe_allow_html=True
)