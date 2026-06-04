import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

st.set_page_config(page_title="Mango Disease Prediction System", page_icon="🥭", layout="wide")

st.title("🥭 Mango Disease Prediction System")
st.markdown("### Weather-Based Mango Disease Early Warning System")

def calculate_week_cyclical(week_number):
    week_sin = np.sin(2 * np.pi * week_number / 52.0)
    week_cos = np.cos(2 * np.pi * week_number / 52.0)
    return week_sin, week_cos

disease = st.selectbox(
    "Select Disease",
    ["Leaf Anthracnose", "Black Banded", "Red Rust", "Die Back", "Sooty Mould"]
)

st.subheader("Current Week Weather Data")

col1, col2 = st.columns(2)

with col1:
    rf = st.number_input("Rainfall (RF, mm)", min_value=0.0, value=24.6)
    rd = st.number_input("Rainy Days (RD)", min_value=0, value=2)
    rh = st.number_input("Relative Humidity (RH, %)", min_value=0.0, max_value=100.0, value=52.4)

with col2:
    t_max = st.number_input("Maximum Temperature (°C)", value=33.6)
    t_min = st.number_input("Minimum Temperature (°C)", value=24.9)

st.subheader("Previous Week Information")

col3, col4 = st.columns(2)

with col3:
    disease_lag1 = st.number_input("Previous Disease Severity", min_value=0.0, value=9.5)
    rf_lag1 = st.number_input("Previous Rainfall (mm)", min_value=0.0, value=25.8)

with col4:
    rh_lag1 = st.number_input("Previous Humidity (%)", min_value=0.0, max_value=100.0, value=49.5)
    rd_lag1 = st.number_input("Previous Rainy Days", min_value=0, value=2)

current_week = datetime.date.today().isocalendar()[1]

std_week = st.number_input(
    "Standard Meteorological Week",
    min_value=1,
    max_value=52,
    value=current_week
)

if st.button("Predict Disease Severity"):

    if t_max <= t_min:
        st.error("Maximum temperature must be greater than minimum temperature.")
        st.stop()

    t_avg = (t_max + t_min) / 2
    t_range = t_max - t_min
    week_sin, week_cos = calculate_week_cyclical(std_week)

    model_files = {
        "Leaf Anthracnose": "leaf_anthracnose_model.pkl",
        "Black Banded": "black_banded_model.pkl",
        "Red Rust": "red_rust_model.pkl",
        "Die Back": "die_back_model.pkl",
        "Sooty Mould": "sooty_mould_model.pkl"
    }

    try:
        model = joblib.load(model_files[disease])

        input_data = pd.DataFrame([[
            rf, rd, rh, t_max, t_min, t_avg, t_range,
            disease_lag1, rf_lag1, rh_lag1, rd_lag1,
            week_sin, week_cos
        ]], columns=[
            "RF", "RD", "RH", "T_MAX", "T_MIN", "T_avg", "T_Range",
            "Disease_lag1", "RF_lag1", "RH_lag1", "RD_lag1",
            "week_sin", "week_cos"
        ])

        prediction = float(model.predict(input_data)[0])

        if prediction < 10:
            risk = "LOW"
            recommendation = "Low disease risk. Routine monitoring is sufficient."
        elif prediction < 20:
            risk = "MODERATE"
            recommendation = "Moderate disease risk. Increase field scouting and monitoring."
        elif prediction < 30:
            risk = "HIGH"
            recommendation = "High disease pressure. Preventive disease management is recommended."
        else:
            risk = "EPIDEMIC"
            recommendation = "Severe outbreak risk. Immediate disease control measures required."

        st.success("Prediction Completed Successfully")

        col5, col6, col7 = st.columns(3)

        with col5:
            st.metric("Predicted Severity", f"{prediction:.2f}")
        with col6:
            st.metric("Risk Category", risk)
        with col7:
            st.metric("Week", std_week)

        st.subheader("Derived Weather Variables")
        st.write(f"Average Temperature (T_avg): **{t_avg:.2f} °C**")
        st.write(f"Temperature Range (T_Range): **{t_range:.2f} °C**")

        st.subheader("Management Recommendation")
        st.info(recommendation)

    except FileNotFoundError:
        st.error(f"Model file not found: {model_files[disease]}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("**Mango Disease Prediction System**")
