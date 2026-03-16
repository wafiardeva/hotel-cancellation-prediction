import streamlit as st
import pandas as pd
import joblib

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

# =========================
# Load Model (cached)
# =========================
@st.cache_resource
def load_model():
    data = joblib.load("hotel_cancellation_model.pkl")
    return data["model"], data["features"]

model, feature_names = load_model()

# =========================
# Header
# =========================
st.title("🏨 Hotel Booking Cancellation Predictor")

st.markdown(
"""
This application predicts the probability of a **hotel booking cancellation**
using a Machine Learning model.

Model used: **LightGBM**
"""
)

st.divider()

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("Booking Information")

lead_time = st.sidebar.slider("Lead Time", 0, 447, 0)

adr = st.sidebar.number_input(
    "Average Daily Rate (ADR)",
    min_value=0.0,
    max_value=252.0,
    value=0.0
)

total_nights = st.sidebar.slider("Total Nights", 1, 69, 1)

total_guests = st.sidebar.slider("Total Guests", 1, 55, 1)

previous_cancellations = st.sidebar.slider(
    "Previous Cancellations",
    0,
    26,
    0
)

deposit_type = st.sidebar.selectbox(
    "Deposit Type",
    ["No Deposit", "Non Refund", "Refundable"]
)

customer_type = st.sidebar.selectbox(
    "Customer Type",
    ["Transient", "Transient-Party", "Contract", "Group"]
)

meal = st.sidebar.selectbox(
    "Meal Type",
    ["BB", "HB", "FB", "SC", "Undefined"]
)

booking_type = st.sidebar.selectbox(
    "Booking Type",
    ["Direct", "Corporate", "Online Travel Agent"]
)

reserved_room_type = st.sidebar.selectbox(
    "Reserved Room Type",
    ["A","B","C","D","E","F","G","H","L"]
)

# =========================
# Prediction Button
# =========================
predict_btn = st.sidebar.button("Predict Cancellation")

# =========================
# Prediction Logic
# =========================
if predict_btn:

    input_data = {
        "lead_time": lead_time,
        "adr": adr,
        "total_nights": total_nights,
        "total_guests": total_guests,
        "previous_cancellations": previous_cancellations,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
        "meal": meal,
        "booking_type": booking_type,
        "reserved_room_type": reserved_room_type
    }

    input_df = pd.DataFrame([input_data])

    # Encoding
    input_df = pd.get_dummies(input_df)

    # Align features
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    # =========================
    # Metrics
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Cancellation Probability",
            value=f"{probability:.2%}"
        )

    with col2:
        if prediction == 1:
            st.metric("Prediction", "Likely Cancel ❌")
        else:
            st.metric("Prediction", "Likely Stay ✅")

    st.progress(float(probability))

    # =========================
    # Risk Level
    # =========================
    if probability < 0.3:
        st.success("Low Cancellation Risk")
    elif probability < 0.7:
        st.warning("Medium Cancellation Risk")
    else:
        st.error("High Cancellation Risk")

    st.divider()

    # =========================
    # Input Summary
    # =========================
    st.subheader("Booking Summary")

    st.dataframe(pd.DataFrame([input_data]))