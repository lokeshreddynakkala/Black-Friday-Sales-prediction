import streamlit as st
import pandas as pd
import joblib
import numpy as np
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])


# --- Page Configuration ---
st.set_page_config(
    page_title="Purchase Predictor",
    page_icon="💸",
    layout="wide"
)

# --- High Visibility & High Contrast CSS ---
st.markdown("""
    <style>
    /* 1. Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* 2. FORCE HEADINGS TO BE BLACK */
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #000000 !important;
        font-weight: 800 !important;
    }

    /* 3. FORCE INPUT LABELS TO BE BLACK */
    label, .stWidgetLabel, [data-testid="stWidgetLabel"] p {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }

    /* 4. METRIC RESULT (Midnight Blue Value, Black Label) */
    [data-testid="stMetricValue"] {
        color: #003366 !important; 
        font-weight: 800;
        font-size: 2.5rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #000000 !important;
        font-size: 1.1rem !important;
        font-weight: 600;
    }

    /* 5. Input Box Styling */
    div[data-testid="stVerticalBlock"] > div:has(div.stNumberInput) {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #ced4da;
    }

    /* 6. Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #000000;
        color: #ffffff;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #333333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load('Black_Friday_model.h5')

try:
    model = load_model()
except:
    st.error("Model 'Black_Friday_model.h5' not found.")
    st.stop()

# --- Visible Heading ---
st.title('🛍️ Black Friday Purchase Predictor')
st.markdown("---")

# --- 4-Column Layout ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("👤 User")
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Age = st.number_input('Age', 12, 100, 25)
    Marital = st.selectbox('Status', ['Single', 'Married'])

with col2:
    st.subheader("📍 Profile")
    Occupation = st.number_input('Occ. ID', 0, 20, 10)
    City = st.selectbox('City Cat', ['A', 'B', 'C'])
    Years = st.number_input('City Years', 0, 10, 2)

with col3:
    st.subheader("📈 Stats")
    U_Mean = st.number_input('User Avg', 0, 500000, 10000)
    U_Count = st.number_input('User Count', 0, 1000, 50)
    P_Mean = st.number_input('Prod Avg', 0, 500000, 10000)

with col4:
    st.subheader("📦 Product")
    C1 = st.number_input('Cat 1', 0, 20, 1)
    C2 = st.number_input('Cat 2', 0, 20, 5)
    P_Count = st.number_input('Prod Count', 0, 1000, 100)

# --- Data Preparation ---
input_df = pd.DataFrame({
    'Gender': [1 if Gender == 'Male' else 0],
    'Age': [Age],
    'Occupation': [Occupation],
    'Stay_In_Current_City_Years': [Years],
    'Marital_Status': [1 if Marital == 'Married' else 0],
    'Product_Category_1': [C1],
    'Product_Category_2': [C2],
    'Product_Category_3': [0],
    'City_Category_B': [1 if City == 'B' else 0],
    'City_Category_C': [1 if City == 'C' else 0],
    'product_mean': [P_Mean],
    'user_mean': [U_Mean],
    'user_count': [U_Count],
    'product_count': [P_Count]
})

st.markdown("---")

if st.button('GENERATE PREDICTION'):
    prediction = model.predict(input_df)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(label="Predicted Total", value=f"${prediction[0]:,.2f}")
    
    with c2:
        # Styled result box
        st.markdown(f"""
            <div style="background-color: #e9ecef; padding: 20px; border-radius: 10px; border-left: 8px solid #003366;">
                <h3 style="margin:0; color: #000000;">Analysis Result</h3>
                <p style="color: #000000; font-size: 1.1rem; margin-top: 10px;">
                    Estimated Purchase: <b>${prediction[0]:,.2f}</b>
                </p>
            </div>
        """, unsafe_allow_html=True)

st.caption("Retail Prediction System ") 

