import streamlit as st
import pandas as pd
import joblib

# =========================
# Load model
# =========================
data = joblib.load("hotel_cancellation_model.pkl")

model = data["model"]
feature_names = data["features"]

# =========================
# Title
# =========================
st.title("Hotel Booking Cancellation Prediction")

st.write("Predict whether a hotel booking will be cancelled.")

st.divider()

# =========================
# User Input
# =========================

lead_time = st.number_input("Lead Time", min_value=0, max_value=500, value=0)

adr = st.number_input("Average Daily Rate (ADR)", min_value=0.0, max_value=252.0, value=0.0)

total_nights = st.number_input("Total Nights", min_value=1, max_value=69, value=1)

total_guests = st.number_input("Total Guests", min_value=1, max_value=55, value=1)

previous_cancellations = st.number_input("Previous Cancellations", min_value=0, max_value=26, value=0)

deposit_type = st.selectbox(
    "Deposit Type",
    ["No Deposit", "Non Refund", "Refundable"]
)

customer_type = st.selectbox(
    "Customer Type",
    ["Transient", "Transient-Party", "Contract", "Group"]
)

meal = st.selectbox(
    "Meal Type",
    ["BB", "HB", "FB", "SC", "Undefined"]
)

booking_type = st.selectbox(
    "Booking Type",
    ["Direct", "Corporate", "Online Travel Agent"]
)

reserved_room_type = st.selectbox(
    "Reserved Room Type",
    ["A","B","C","D","E","F","G","H","L"]
)

# =========================
# Prediction
# =========================

if st.button("Predict Cancellation"):

    # Create dataframe from input
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

    # Apply encoding
    input_df = pd.get_dummies(input_df)

    # Align features with training
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"Booking likely to CANCEL")
    else:
        st.success(f"Booking likely to NOT cancel")

    st.write(f"Cancellation Probability: **{probability:.2%}**")
