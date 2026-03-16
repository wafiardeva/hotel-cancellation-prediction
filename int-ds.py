import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Hotel Cancellation Intelligence",
    page_icon="🏨",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    data = joblib.load("hotel_cancellation_model.pkl")
    return data["model"], data["features"]

model, feature_names = load_model()

# =========================
# HEADER
# =========================
st.title("🏨 Hotel Booking Cancellation Intelligence")

st.markdown(
"""
Machine Learning dashboard to predict **hotel booking cancellations**  
and analyze risk factors using a **LightGBM classification model**.
"""
)

st.divider()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Booking Information")

lead_time = st.sidebar.slider("Lead Time (days before arrival)", 0, 447, 0)

adr = st.sidebar.number_input(
    "ADR (Average Daily Rate)",
    min_value=0.0,
    max_value=252.0,
    value=0.0
)

total_nights = st.sidebar.slider("Total Nights", 1, 69, 1)

total_guests = st.sidebar.slider("Total Guests", 1, 55, 1)

previous_cancellations = st.sidebar.slider(
    "Previous Cancellations",
    0, 26, 0
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
    "Meal",
    ["BB", "HB", "FB", "SC", "Undefined"]
)

booking_type = st.sidebar.selectbox(
    "Booking Channel",
    ["Direct", "Corporate", "Online Travel Agent"]
)

reserved_room_type = st.sidebar.selectbox(
    "Reserved Room Type",
    ["A","B","C","D","E","F","G","H","L"]
)

predict = st.sidebar.button("Predict Cancellation")

# =========================
# PREDICTION
# =========================
if predict:

    with st.spinner("Running prediction model..."):

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

        # Align columns
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Model prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

    # =========================
    # KPI METRICS
    # =========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Cancellation Probability", f"{probability:.2%}")

    with col2:
        if prediction == 1:
            st.metric("Prediction", "Likely Cancel ❌")
        else:
            st.metric("Prediction", "Likely Stay ✅")

    with col3:
        if probability < 0.3:
            risk = "Low"
        elif probability < 0.7:
            risk = "Medium"
        else:
            risk = "High"

        st.metric("Risk Level", risk)

    # =========================
    # GAUGE CHART
    # =========================
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Cancellation Risk"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "blue"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

    st.divider()

    # =========================
    # SHAP EXPLANATION
    # =========================
    st.subheader("Prediction Explanation (SHAP)")

    with st.spinner("Calculating SHAP explanation..."):

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        shap_df = pd.DataFrame({
            "feature": input_df.columns,
            "impact": shap_values[0]
        })

        shap_df = shap_df.sort_values(
            "impact",
            key=abs,
            ascending=False
        ).head(10)

        fig = px.bar(
            shap_df,
            x="impact",
            y="feature",
            orientation="h",
            title="Top Factors Influencing Prediction"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # =========================
    # BOOKING SUMMARY
    # =========================
    st.subheader("Booking Summary")

    st.dataframe(pd.DataFrame([input_data]))

# =========================
# GLOBAL FEATURE IMPORTANCE
# =========================
st.divider()

st.subheader("Global Feature Importance")

importance = model.feature_importances_

fi = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

fi = fi.sort_values(
    "importance",
    ascending=False
).head(15)

fig2 = px.bar(
    fi,
    x="importance",
    y="feature",
    orientation="h",
    title="Top Global Features"
)

st.plotly_chart(fig2, use_container_width=True)