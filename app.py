from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Chun_prediction_model.h5"


st.set_page_config(
    page_title="ChurnGuard | Customer Retention Tool",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    [data-testid="stMetricValue"] {
        color: #1c4966;
    }
    [data-testid="stMetricLabel"] {
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error(f"Error loading model: {exc}")
    st.stop()


with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3121/3121768.png", width=100)
    st.title("ChurnGuard AI")
    st.info("This tool uses machine learning to predict the likelihood of a customer leaving the bank.")
    st.markdown("---")
    st.markdown("### Model Info")
    st.text("Type: Gradient Boosting Classifier")


st.title("Customer Churn Analytics")
st.markdown("Fill in the customer profile below to generate a retention risk assessment.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)

with col2:
    st.subheader("Financial Profile")
    credit_score = st.number_input("Credit Score", 300, 850, 650)
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=500.0)
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=75000.0, step=500.0)

with col3:
    st.subheader("Product Usage")
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_credit_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
    is_active_member = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True)


gender_map = {"Male": [1, 0], "Female": [0, 1]}
g_m, g_f = gender_map[gender]

geo_map = {"France": [1, 0], "Germany": [0, 1], "Spain": [0, 0]}
geo_f, geo_g = geo_map[geography]

input_df = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "EstimatedSalary": [estimated_salary],
        "HasCrCard": [1 if has_credit_card == "Yes" else 0],
        "IsActiveMember": [1 if is_active_member == "Yes" else 0],
        "Gender_Male": [g_m],
        "Gender_Female": [g_f],
        "Geography_France": [geo_f],
        "Geography_Germany": [geo_g],
    }
)


st.markdown("---")
if st.button("Analyze Retention Risk"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[0][1]

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.subheader("Risk Assessment")
        if prediction[0] == 0:
            st.success("### LOW RISK")
            st.write("This customer is likely to stay with the bank.")
        else:
            st.warning("### HIGH RISK")
            st.write("This customer shows high probability of churning.")

    with res_col2:
        st.subheader("Churn Probability")
        churn_percentage = prediction_proba * 100
        st.markdown(
            f"""
            <div style="
                background-color: #e1effe;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #1c4966;
                text-align: center;">
                <p style="color: #1c4966; font-size: 18px; margin-bottom: 5px; font-weight: bold;">
                    Calculated Churn Risk
                </p>
                <h2 style="color: #000000; margin-top: 0px;">
                    {churn_percentage:.2f}%
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(prediction_proba)
