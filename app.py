import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="EV Range Predictor", page_icon="⚡", layout="centered")

# ── Train model from CSV (cached so it only runs once) ─────
@st.cache_resource
def train_model():
    df = pd.read_csv("EV_Range_Prediction_Dataset_2000.csv")

    # Drop missing values
    df = df.dropna(subset=['battery_soc', 'power_consumption', 'speed',
                            'remaining_range', 'regen_braking'])

    data = df.copy()

    # Encode categorical columns
    le_driving = LabelEncoder()
    le_traffic = LabelEncoder()

    data['driving_style']   = le_driving.fit_transform(data['driving_style'])
    data['traffic_density'] = le_traffic.fit_transform(data['traffic_density'])

    features = [
        'battery_soc', 'speed', 'power_consumption', 'battery_health',
        'road_gradient', 'temperature', 'regen_braking',
        'driving_style', 'traffic_density'
    ]

    X = data[features]
    y = data['remaining_range']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Return model + label classes so we can encode user input the same way
    return model, le_driving.classes_.tolist(), le_traffic.classes_.tolist()

model, driving_styles, traffic_densities = train_model()

# ── UI ─────────────────────────────────────────────────────
st.title("⚡ EV Remaining Range Predictor")
st.markdown("Enter your vehicle and driving conditions to estimate the remaining range.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    battery_soc    = st.slider("🔋 Battery SoC (%)", 0, 100, 75)
    battery_health = st.slider("🏥 Battery Health (%)", 0, 100, 90)
    temperature    = st.slider("🌡️ Temperature (°C)", -20, 50, 25)
    road_gradient  = st.slider("⛰️ Road Gradient (%)", -20, 20, 0)

with col2:
    speed           = st.number_input("🚗 Speed (km/h)", min_value=0.0, max_value=200.0, value=60.0)
    power_consumption = st.number_input("⚡ Power Consumption (kW)", min_value=0.0, max_value=100.0, value=15.0)
    regen_braking   = st.number_input("♻️ Regen Braking (kW)", min_value=0.0, max_value=50.0, value=5.0)
    driving_style   = st.selectbox("🎮 Driving Style", driving_styles)
    traffic_density = st.selectbox("🚦 Traffic Density", traffic_densities)

st.divider()

# ── Encode user input ───────────────────────────────────────
def encode_input():
    ds = driving_styles.index(driving_style)
    td = traffic_densities.index(traffic_density)

    return np.array([[
        battery_soc, speed, power_consumption, battery_health,
        road_gradient, temperature, regen_braking, ds, td
    ]])

# ── Predict ─────────────────────────────────────────────────
if st.button("🔍 Predict Remaining Range", use_container_width=True):
    features = encode_input()
    prediction = model.predict(features)[0]

    st.success(f"### 🛣️ Estimated Remaining Range: **{prediction:.1f} km**")

    st.markdown("---")
    st.subheader("📊 Key Insights")

    insights = []
    if battery_soc < 20:
        insights.append("🔴 **Low battery SoC** — charge soon for safer range.")
    if power_consumption > 30:
        insights.append("⚠️ **High power consumption** is reducing your range significantly.")
    if temperature < 5:
        insights.append("🥶 **Cold temperature** reduces battery efficiency and range.")
    if road_gradient > 10:
        insights.append("⛰️ **Steep road gradient** increases energy demand.")
    if driving_style.lower() == "aggressive":
        insights.append("🏎️ **Aggressive driving** drains the battery faster.")
    if not insights:
        insights.append("✅ Conditions look good for an efficient drive!")

    for i in insights:
        st.markdown(i)

st.markdown("---")
st.caption("Model: Random Forest Regressor | Built with scikit-learn & Streamlit")
