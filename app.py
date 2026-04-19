import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ── Page config ────────────────────────────────────────────
st.set_page_config(page_title="EV Range Predictor", page_icon="⚡", layout="wide")

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #0a0f1e;
    color: #e0e8ff;
}

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
}

/* Hide default streamlit elements */
#MainMenu, footer, header {visibility: hidden;}

/* Hero banner */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(180deg, rgba(0,200,255,0.05) 0%, transparent 100%);
    border-bottom: 1px solid rgba(0,200,255,0.1);
    margin-bottom: 2rem;
}

.hero h1 {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00c8ff, #00ff9d, #00c8ff);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s infinite linear;
    margin-bottom: 0.3rem;
}

@keyframes shimmer {
    0% { background-position: 0% }
    100% { background-position: 200% }
}

.hero p {
    color: #7a9bbf;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.1em;
}

/* Section labels */
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #00c8ff;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-left: 0.5rem;
    border-left: 2px solid #00c8ff;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(0,200,255,0.07), rgba(0,255,157,0.03));
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    transition: border-color 0.3s;
}

.metric-card:hover {
    border-color: rgba(0,200,255,0.5);
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, rgba(0,255,157,0.1), rgba(0,200,255,0.05));
    border: 2px solid #00ff9d;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 0 40px rgba(0,255,157,0.15);
}

.result-km {
    font-family: 'Orbitron', monospace;
    font-size: 4rem;
    font-weight: 900;
    color: #00ff9d;
    line-height: 1;
}

.result-label {
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    color: #7a9bbf;
    margin-top: 0.5rem;
    text-transform: uppercase;
}

/* Gauge bar */
.gauge-container {
    background: rgba(255,255,255,0.05);
    border-radius: 50px;
    height: 12px;
    margin: 0.5rem 0;
    overflow: hidden;
}

.gauge-fill {
    height: 100%;
    border-radius: 50px;
    transition: width 1s ease;
}

/* Insight cards */
.insight-card {
    background: rgba(255,200,0,0.05);
    border: 1px solid rgba(255,200,0,0.2);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
}

.insight-good {
    background: rgba(0,255,157,0.05);
    border-color: rgba(0,255,157,0.2);
}

/* Slider labels */
.stSlider label {
    font-family: 'Exo 2', sans-serif !important;
    color: #7a9bbf !important;
    font-size: 0.85rem !important;
}

/* Input labels */
.stNumberInput label, .stSelectbox label {
    color: #7a9bbf !important;
    font-size: 0.85rem !important;
}

/* Button */
.stButton > button {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.15em !important;
    background: linear-gradient(90deg, #00c8ff, #00ff9d) !important;
    color: #0a0f1e !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.8rem 2rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    box-shadow: 0 4px 20px rgba(0,200,255,0.3) !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

/* Divider */
hr {
    border-color: rgba(0,200,255,0.1) !important;
}

/* Feature importance bar */
.feat-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
}

.feat-name {
    width: 160px;
    color: #7a9bbf;
    flex-shrink: 0;
}

.feat-bar-bg {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}

.feat-bar {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #00c8ff, #00ff9d);
}

.feat-val {
    width: 45px;
    text-align: right;
    color: #00c8ff;
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
}
</style>
""", unsafe_allow_html=True)

# ── Train model ─────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv("EV_Range_Prediction_Dataset_2000.csv")
    df = df.dropna(subset=['battery_soc', 'power_consumption', 'speed',
                            'remaining_range', 'regen_braking'])
    data = df.copy()

    le_driving = LabelEncoder()
    le_traffic = LabelEncoder()
    data['driving_style']   = le_driving.fit_transform(data['driving_style'])
    data['traffic_density'] = le_traffic.fit_transform(data['traffic_density'])

    features = ['battery_soc', 'speed', 'power_consumption', 'battery_health',
                'road_gradient', 'temperature', 'regen_braking',
                'driving_style', 'traffic_density']

    X = data[features]
    y = data['remaining_range']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    importances = dict(zip(features, model.feature_importances_))
    return model, le_driving.classes_.tolist(), le_traffic.classes_.tolist(), importances

model, driving_styles, traffic_densities, importances = train_model()

# ── Hero ────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>⚡ EV RANGE PREDICTOR</h1>
    <p>REAL-TIME REMAINING RANGE ESTIMATION · RANDOM FOREST · 2000 DATA POINTS</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ──────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="section-label">⚡ Battery & Vehicle Status</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        battery_soc = st.slider("🔋 Battery SoC (%)", 0, 100, 75)
        # Live color bar
        soc_color = "#00ff9d" if battery_soc > 50 else "#ffaa00" if battery_soc > 20 else "#ff4444"
        st.markdown(f"""
        <div class="gauge-container">
            <div class="gauge-fill" style="width:{battery_soc}%; background:{soc_color};"></div>
        </div>""", unsafe_allow_html=True)

    with col2:
        battery_health = st.slider("🏥 Battery Health (%)", 0, 100, 90)
        health_color = "#00ff9d" if battery_health > 70 else "#ffaa00" if battery_health > 40 else "#ff4444"
        st.markdown(f"""
        <div class="gauge-container">
            <div class="gauge-fill" style="width:{battery_health}%; background:{health_color};"></div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:1.5rem;">🚗 Driving Conditions</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        speed             = st.number_input("🚗 Speed (km/h)", 0.0, 200.0, 60.0, step=1.0)
        power_consumption = st.number_input("⚡ Power Consumption (kW)", 0.0, 100.0, 15.0, step=0.5)
        regen_braking     = st.number_input("♻️ Regen Braking (kW)", 0.0, 50.0, 5.0, step=0.5)

    with col4:
        temperature   = st.slider("🌡️ Temperature (°C)", -20, 50, 25)
        road_gradient = st.slider("⛰️ Road Gradient (%)", -20, 20, 0)
        driving_style   = st.selectbox("🎮 Driving Style", driving_styles)
        traffic_density = st.selectbox("🚦 Traffic Density", traffic_densities)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ CALCULATE REMAINING RANGE")

with right:
    st.markdown('<div class="section-label">📊 Model Intelligence</div>', unsafe_allow_html=True)

    # Feature importance chart
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    max_imp = max(v for _, v in sorted_imp)

    feat_html = ""
    for feat, val in sorted_imp:
        bar_width = int((val / max_imp) * 100)
        feat_html += f"""
        <div class="feat-row">
            <div class="feat-name">{feat.replace('_', ' ').title()}</div>
            <div class="feat-bar-bg"><div class="feat-bar" style="width:{bar_width}%"></div></div>
            <div class="feat-val">{val:.3f}</div>
        </div>"""

    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:0.75rem;color:#7a9bbf;letter-spacing:0.1em;margin-bottom:0.8rem;">
            FEATURE IMPORTANCE (RANDOM FOREST)
        </div>
        {feat_html}
    </div>
    """, unsafe_allow_html=True)

    # Live condition indicators
    st.markdown('<div class="section-label" style="margin-top:1rem;">🟢 Live Condition Check</div>', unsafe_allow_html=True)

    def condition_badge(label, ok, ok_text, warn_text):
        color = "#00ff9d" if ok else "#ff6b6b"
        text  = ok_text if ok else warn_text
        return f'<div class="insight-card {"insight-good" if ok else ""}">{"✅" if ok else "⚠️"} <b style="color:{color}">{label}:</b> {text}</div>'

    st.markdown(
        condition_badge("Battery", battery_soc > 20, f"{battery_soc}% — Good", f"{battery_soc}% — Critically Low") +
        condition_badge("Temperature", -5 < temperature < 40, f"{temperature}°C — Optimal", f"{temperature}°C — Affects Range") +
        condition_badge("Power Load", power_consumption < 30, f"{power_consumption} kW — Efficient", f"{power_consumption} kW — High Draw") +
        condition_badge("Road", abs(road_gradient) < 10, f"{road_gradient}% — Flat", f"{road_gradient}% — Steep Gradient"),
        unsafe_allow_html=True
    )

    # Result
    if predict_btn:
        ds = driving_styles.index(driving_style)
        td = traffic_densities.index(traffic_density)
        features_input = np.array([[battery_soc, speed, power_consumption, battery_health,
                                     road_gradient, temperature, regen_braking, ds, td]])
        prediction = model.predict(features_input)[0]

        # Range colour
        range_color = "#00ff9d" if prediction > 150 else "#ffaa00" if prediction > 60 else "#ff4444"

        st.markdown(f"""
        <div class="result-box" style="border-color:{range_color}; box-shadow: 0 0 40px {range_color}33;">
            <div style="font-family:'Orbitron',monospace;font-size:0.7rem;letter-spacing:0.2em;color:#7a9bbf;margin-bottom:0.5rem;">
                ESTIMATED REMAINING RANGE
            </div>
            <div class="result-km" style="color:{range_color};">{prediction:.1f}</div>
            <div class="result-label">KILOMETERS</div>
        </div>
        """, unsafe_allow_html=True)

        # Insights
        insights = []
        if battery_soc < 20:
            insights.append("🔴 Low SoC — charge soon.")
        if power_consumption > 30:
            insights.append("⚠️ High power draw shortening range.")
        if temperature < 5:
            insights.append("🥶 Cold temps reduce battery efficiency.")
        if road_gradient > 10:
            insights.append("⛰️ Steep gradient increases energy use.")
        if driving_style.lower() == "aggressive":
            insights.append("🏎️ Aggressive style drains battery faster.")
        if regen_braking > 10:
            insights.append("♻️ Good regen braking — recovering energy!")
        if not insights:
            insights.append("✅ All conditions optimal for maximum range!")

        for ins in insights:
            st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="text-align:center;color:#2a3a5a;font-size:0.75rem;font-family:\'Orbitron\',monospace;letter-spacing:0.1em;">RANDOM FOREST REGRESSOR · SCIKIT-LEARN · STREAMLIT · © 2026</p>', unsafe_allow_html=True)