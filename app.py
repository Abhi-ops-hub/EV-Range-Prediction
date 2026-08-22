import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="EV Range Predictor", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body {
    font-family: 'Exo 2', sans-serif !important;
    background-color: #0a0f1e !important;
    color: #e0e8ff !important;
}

.stApp, .stApp > div, section.main, section.main > div, .block-container {
    background-color: #0a0f1e !important;
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%) !important;
    color: #e0e8ff !important;
}

section[data-testid="stSidebar"] { background-color: #0d1b2a !important; }
#MainMenu, footer, header { visibility: hidden !important; }
p, div, span, label { color: #e0e8ff !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(0,200,255,0.05) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(0,200,255,0.15) !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    color: #7a9bbf !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(0,200,255,0.2), rgba(0,255,157,0.1)) !important;
    color: #00c8ff !important;
    border-bottom: 2px solid #00c8ff !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 0 !important; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(180deg, rgba(0,200,255,0.07) 0%, transparent 100%);
    border-bottom: 1px solid rgba(0,200,255,0.15);
    margin-bottom: 2rem;
    border-radius: 0 0 16px 16px;
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
    0%   { background-position: 0% }
    100% { background-position: 200% }
}
.hero p { color: #7a9bbf !important; font-size: 1rem; font-weight: 300; letter-spacing: 0.1em; }

/* ── Metric Pills ── */
.metric-pill {
    background: linear-gradient(135deg, rgba(0,200,255,0.1), rgba(0,255,157,0.05));
    border: 1px solid rgba(0,200,255,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-pill:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,200,255,0.2); }
.metric-pill .val { font-family: 'Orbitron', monospace; font-size: 1.6rem; font-weight: 900; color: #00ff9d; }
.metric-pill .lbl { font-size: 0.7rem; letter-spacing: 0.15em; color: #7a9bbf !important; text-transform: uppercase; margin-top: 0.2rem; }

/* ── Section Labels ── */
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #00c8ff !important;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding-left: 0.5rem;
    border-left: 2px solid #00c8ff;
}

/* ── Gauge Bars ── */
.gauge-container { background: rgba(255,255,255,0.07); border-radius: 50px; height: 8px; margin: 0.3rem 0 1rem; overflow: hidden; }
.gauge-fill { height: 100%; border-radius: 50px; transition: width 0.5s ease; }

/* ── Insight Cards ── */
.insight-card {
    background: rgba(255,200,0,0.05);
    border: 1px solid rgba(255,200,0,0.2);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin: 0.3rem 0;
    font-size: 0.88rem;
    transition: transform 0.15s ease;
}
.insight-card:hover { transform: translateX(3px); }
.insight-good { background: rgba(0,255,157,0.05); border-color: rgba(0,255,157,0.2); }

/* ── Result Box ── */
.result-box {
    background: linear-gradient(135deg, rgba(0,255,157,0.08), rgba(0,200,255,0.05));
    border: 2px solid #00ff9d;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 0 40px rgba(0,255,157,0.15);
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-km { font-family: 'Orbitron', monospace; font-size: 4rem; font-weight: 900; color: #00ff9d; line-height: 1; }
.result-label { font-size: 0.85rem; letter-spacing: 0.2em; color: #7a9bbf !important; margin-top: 0.5rem; text-transform: uppercase; }

/* ── Chart Card ── */
.chart-card {
    background: rgba(0,200,255,0.03);
    border: 1px solid rgba(0,200,255,0.12);
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 1.2rem;
    transition: box-shadow 0.2s ease;
}
.chart-card:hover { box-shadow: 0 4px 24px rgba(0,200,255,0.12); }
.chart-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    color: #00c8ff !important;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* ── Stat Cards ── */
.stat-card {
    background: linear-gradient(135deg, rgba(0,255,157,0.07), rgba(0,200,255,0.04));
    border: 1px solid rgba(0,255,157,0.18);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.stat-card .s-val { font-family: 'Orbitron', monospace; font-size: 1.3rem; font-weight: 700; color: #00ff9d; }
.stat-card .s-lbl { font-size: 0.68rem; letter-spacing: 0.12em; color: #7a9bbf !important; text-transform: uppercase; margin-top: 0.15rem; }

/* ── Input Widgets ── */
.stSlider label, .stNumberInput label, .stSelectbox label {
    color: #7a9bbf !important; font-size: 0.85rem !important; font-weight: 400 !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #00c8ff !important; border-color: #00c8ff !important;
}
input[type="number"], .stSelectbox > div > div {
    background-color: rgba(0,200,255,0.05) !important;
    border-color: rgba(0,200,255,0.2) !important;
    color: #e0e8ff !important;
    border-radius: 8px !important;
}

/* ── Predict Button ── */
.stButton > button {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.15em !important;
    background: linear-gradient(90deg, #00c8ff, #00ff9d) !important;
    color: #0a0f1e !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.85rem 2rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(0,200,255,0.3) !important;
    transition: opacity 0.2s ease, transform 0.2s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

.stDataFrame, iframe { border-radius: 10px !important; }
hr { border-color: rgba(0,200,255,0.1) !important; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# Load data & train model (cached)
# ────────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    df_raw = pd.read_csv("EV_Range_Prediction_Dataset_2000.csv")
    df = df_raw.dropna(subset=['battery_soc', 'power_consumption', 'speed',
                                'remaining_range', 'regen_braking'])
    data = df.copy()
    le_driving = LabelEncoder()
    le_traffic = LabelEncoder()
    data['driving_style']   = le_driving.fit_transform(data['driving_style'])
    data['traffic_density'] = le_traffic.fit_transform(data['traffic_density'])

    features = ['battery_soc', 'speed', 'power_consumption', 'battery_health',
                'road_gradient', 'temperature', 'regen_braking',
                'driving_style', 'traffic_density']
    X, y = data[features], data['remaining_range']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    importances = dict(zip(features, model.feature_importances_))
    return model, le_driving.classes_.tolist(), le_traffic.classes_.tolist(), importances, df_raw

model, driving_styles, traffic_densities, importances, df_raw = load_and_train()

# ────────────────────────────────────────────────────────────
# Hero
# ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>⚡ EV RANGE PREDICTOR</h1>
    <p>REAL-TIME REMAINING RANGE ESTIMATION · RANDOM FOREST · 2000 DATA POINTS</p>
</div>
""", unsafe_allow_html=True)

# ── Model Metrics Row ──
m1, m2, m3, m4 = st.columns(4)
for col, val, lbl in [
    (m1, "98%",  "R² Accuracy"),
    (m2, "5.98", "MAE (km)"),
    (m3, "100",  "Estimators"),
    (m4, "2000", "Data Points"),
]:
    with col:
        st.markdown(f'<div class="metric-pill"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# Tabs
# ────────────────────────────────────────────────────────────
tab_predictor, tab_explorer = st.tabs(["⚡  RANGE PREDICTOR", "📊  DATA EXPLORER"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — RANGE PREDICTOR
# ══════════════════════════════════════════════════════════════
with tab_predictor:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="section-label">⚡ Battery &amp; Vehicle Status</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            battery_soc = st.slider("🔋 Battery SoC (%)", 0, 100, 75)
            soc_color = "#00ff9d" if battery_soc > 50 else "#ffaa00" if battery_soc > 20 else "#ff4444"
            st.markdown(
                f'<div class="gauge-container"><div class="gauge-fill" style="width:{battery_soc}%;background:{soc_color};"></div></div>',
                unsafe_allow_html=True)
        with col2:
            battery_health = st.slider("🏥 Battery Health (%)", 0, 100, 90)
            health_color = "#00ff9d" if battery_health > 70 else "#ffaa00" if battery_health > 40 else "#ff4444"
            st.markdown(
                f'<div class="gauge-container"><div class="gauge-fill" style="width:{battery_health}%;background:{health_color};"></div></div>',
                unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1rem;">🚗 Driving Conditions</div>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            speed             = st.number_input("🚗 Speed (km/h)",          0.0, 200.0, 60.0,  step=1.0)
            power_consumption = st.number_input("⚡ Power Consumption (kW)", 0.0, 100.0, 15.0,  step=0.5)
            regen_braking     = st.number_input("♻️ Regen Braking (kW)",    0.0,  50.0,  5.0,  step=0.5)
        with col4:
            temperature     = st.slider("🌡️ Temperature (°C)",  -20, 50, 25)
            road_gradient   = st.slider("⛰️ Road Gradient (%)",  -20, 20,  0)
            driving_style   = st.selectbox("🎮 Driving Style",   driving_styles)
            traffic_density = st.selectbox("🚦 Traffic Density", traffic_densities)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ CALCULATE REMAINING RANGE")

    with right:
        st.markdown('<div class="section-label">📊 Feature Importance</div>', unsafe_allow_html=True)
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        imp_df = pd.DataFrame(sorted_imp, columns=["Feature", "Importance"])
        imp_df["Feature"] = imp_df["Feature"].str.replace("_", " ").str.title()
        st.dataframe(
            imp_df.style.bar(subset=["Importance"], color="#00c8ff").format({"Importance": "{:.4f}"}),
            use_container_width=True, hide_index=True, height=320)

        st.markdown('<div class="section-label" style="margin-top:1rem;">🟢 Live Condition Check</div>', unsafe_allow_html=True)
        conditions = [
            ("🔋 Battery",      battery_soc > 20,
             f"{battery_soc}% — Good charge level",
             f"{battery_soc}% — Critically low, charge now!"),
            ("🌡️ Temperature",  -5 < temperature < 40,
             f"{temperature}°C — Optimal range",
             f"{temperature}°C — Affects battery performance"),
            ("⚡ Power Load",   power_consumption < 30,
             f"{power_consumption} kW — Efficient",
             f"{power_consumption} kW — High draw, range drops"),
            ("⛰️ Road",         abs(road_gradient) < 10,
             f"{road_gradient}% — Fairly flat",
             f"{road_gradient}% — Steep, consuming more energy"),
        ]
        for label, ok, good_text, bad_text in conditions:
            icon  = "✅" if ok else "⚠️"
            css   = "insight-good" if ok else ""
            color = "#00ff9d" if ok else "#ff6b6b"
            text  = good_text if ok else bad_text
            st.markdown(
                f'<div class="insight-card {css}">{icon} <b style="color:{color}">{label}:</b> {text}</div>',
                unsafe_allow_html=True)

        # ── Prediction Result ──
        if predict_btn:
            ds = driving_styles.index(driving_style)
            td = traffic_densities.index(traffic_density)
            feat_input = np.array([[battery_soc, speed, power_consumption, battery_health,
                                     road_gradient, temperature, regen_braking, ds, td]])
            prediction = model.predict(feat_input)[0]
            range_color = "#00ff9d" if prediction > 150 else "#ffaa00" if prediction > 60 else "#ff4444"

            st.markdown(f"""
            <div class="result-box" style="border-color:{range_color};box-shadow:0 0 40px {range_color}33;">
                <div style="font-family:'Orbitron',monospace;font-size:0.7rem;letter-spacing:0.2em;color:#7a9bbf;margin-bottom:0.5rem;">
                    ESTIMATED REMAINING RANGE
                </div>
                <div class="result-km" style="color:{range_color};">{prediction:.1f}</div>
                <div class="result-label">KILOMETERS</div>
            </div>
            """, unsafe_allow_html=True)

            insights = []
            if battery_soc < 20:                          insights.append("🔴 Low SoC — please charge soon.")
            if power_consumption > 30:                    insights.append("⚠️ High power draw is reducing range.")
            if temperature < 5:                           insights.append("🥶 Cold temperature reduces battery efficiency.")
            if road_gradient > 10:                        insights.append("⛰️ Steep gradient is consuming more energy.")
            if driving_style.strip().lower() == "aggressive": insights.append("🏎️ Aggressive driving drains battery faster.")
            if regen_braking > 10:                        insights.append("♻️ Great regen braking — recovering energy!")
            if not insights:                              insights.append("✅ All conditions optimal for maximum range!")

            st.markdown('<div class="section-label" style="margin-top:1rem;">💡 Insights</div>', unsafe_allow_html=True)
            for ins in insights:
                st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════
with tab_explorer:

    # ── Dataset Overview Stats ──
    st.markdown('<div class="section-label">🗃️ Dataset Overview</div>', unsafe_allow_html=True)

    s1, s2, s3, s4, s5 = st.columns(5)
    stat_data = [
        (s1, str(len(df_raw)),     "Total Records"),
        (s2, str(df_raw.shape[1]), "Features"),
        (s3, f"{df_raw['remaining_range'].mean():.1f} km", "Avg Remaining Range"),
        (s4, f"{df_raw['battery_soc'].mean():.1f}%",       "Avg Battery SoC"),
        (s5, f"{df_raw['power_consumption'].mean():.1f} kW", "Avg Power Draw"),
    ]
    for col, val, lbl in stat_data:
        with col:
            st.markdown(f'<div class="stat-card"><div class="s-val">{val}</div><div class="s-lbl">{lbl}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Raw Data Preview ──
    with st.expander("📋  Preview Raw Dataset", expanded=False):
        st.dataframe(df_raw.head(50), use_container_width=True, hide_index=True)

    # ── Descriptive Statistics ──
    with st.expander("📈  Descriptive Statistics", expanded=False):
        st.dataframe(df_raw.describe().round(3), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────
    # Analysis Charts — grouped by theme
    # ────────────────────────────────────────────────────────

    # Group 1: Battery Analysis
    st.markdown('<div class="section-label">🔋 Battery Analysis</div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        st.markdown('<div class="chart-card"><div class="chart-title">SoC vs Remaining Range</div>', unsafe_allow_html=True)
        try:
            st.image("soc vs remaining range.png", use_container_width=True)
        except Exception:
            st.warning("Image not found: soc vs remaining range.png")
        st.markdown('</div>', unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="chart-card"><div class="chart-title">Battery SOC vs Power Consumption</div>', unsafe_allow_html=True)
        try:
            st.image("Battery SOC vs Power Consumption.png", use_container_width=True)
        except Exception:
            st.warning("Image not found: Battery SOC vs Power Consumption.png")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Group 2: Power Consumption
    st.markdown('<div class="section-label">⚡ Power Consumption Analysis</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.markdown('<div class="chart-card"><div class="chart-title">Speed vs Power Consumption</div>', unsafe_allow_html=True)
        try:
            st.image("Speed vs Power Consumption.png", use_container_width=True)
        except Exception:
            st.warning("Image not found: Speed vs Power Consumption.png")
        st.markdown('</div>', unsafe_allow_html=True)

    with p2:
        st.markdown('<div class="chart-card"><div class="chart-title">Power Consumption at Road Gradient</div>', unsafe_allow_html=True)
        try:
            st.image("Power consumption at road gradient.png", use_container_width=True)
        except Exception:
            st.warning("Image not found: Power consumption at road gradient.png")
        st.markdown('</div>', unsafe_allow_html=True)

    p3, p4 = st.columns(2)
    with p3:
        st.markdown('<div class="chart-card"><div class="chart-title">Top Power Consumption at Speed</div>', unsafe_allow_html=True)
        try:
            st.image("Top power consumption at speed.png", use_container_width=True)
        except Exception:
            st.warning("Image not found: Top power consumption at speed.png")
        st.markdown('</div>', unsafe_allow_html=True)

    with p4:
        st.markdown('<div class="chart-card"><div class="chart-title">Power Consumption by Driving Style</div>', unsafe_allow_html=True)
        try:
            st.image("power consumption by driving style.png", use_container_width=True)
        except Exception:
            st.warning("Image not found: power consumption by driving style.png")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Group 3: Driving Style & Traffic
    st.markdown('<div class="section-label">🚗 Driving Style &amp; Traffic</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.markdown('<div class="chart-card"><div class="chart-title">Driving Style vs Avg Power Consumption</div>', unsafe_allow_html=True)
        try:
            st.image("Driving Style vs Avg Power Consumption.png", use_container_width=True)
        except Exception:
            st.warning("Image not found: Driving Style vs Avg Power Consumption.png")
        st.markdown('</div>', unsafe_allow_html=True)

    with d2:
        st.markdown('<div class="chart-card"><div class="chart-title">Range Impact by Traffic Density</div>', unsafe_allow_html=True)
        try:
            st.image("Range impact by traffic density.png", use_container_width=True)
        except Exception:
            st.warning("Image not found: Range impact by traffic density.png")
        st.markdown('</div>', unsafe_allow_html=True)

    d3, _ = st.columns(2)
    with d3:
        st.markdown('<div class="chart-card"><div class="chart-title">Traffic Density Impact on Range</div>', unsafe_allow_html=True)
        try:
            st.image("Traffic density impact.png", use_container_width=True)
        except Exception:
            st.warning("Image not found: Traffic density impact.png")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Group 4: Environmental
    st.markdown('<div class="section-label">🌡️ Environmental Factors</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        st.markdown('<div class="chart-card"><div class="chart-title">Temperature vs Remaining Range</div>', unsafe_allow_html=True)
        try:
            st.image("Temperature vs remaining range.png", use_container_width=True)
        except Exception:
            st.warning("Image not found: Temperature vs remaining range.png")
        st.markdown('</div>', unsafe_allow_html=True)

    with e2:
        st.markdown('<div class="chart-card"><div class="chart-title">Regenerative Braking vs Remaining Range</div>', unsafe_allow_html=True)
        try:
            st.image("regenerative braking vs remaining_range.png", use_container_width=True)
        except Exception:
            st.warning("Image not found: regenerative braking vs remaining_range.png")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Group 5: Correlations & Overview
    st.markdown('<div class="section-label">🔬 Correlation &amp; Feature Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="chart-card"><div class="chart-title">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        try:
            st.image("Feature Correlation Heatmap.png", use_container_width=True)
        except Exception:
            st.warning("Image not found: Feature Correlation Heatmap.png")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card"><div class="chart-title">Remaining Range Over Time</div>', unsafe_allow_html=True)
        try:
            st.image("time series over time.png", use_container_width=True)
        except Exception:
            st.warning("Image not found: time series over time.png")
        st.markdown('</div>', unsafe_allow_html=True)

    # Most affect range — full width
    st.markdown('<div class="chart-card"><div class="chart-title">Feature Pairplot — Most Affect on Range</div>', unsafe_allow_html=True)
    try:
        st.image("most affect range.png", use_container_width=True)
    except Exception:
        st.warning("Image not found: most affect range.png")
    st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#2a3a5a;font-size:0.75rem;'
    'font-family:Orbitron,monospace;letter-spacing:0.1em;">'
    'RANDOM FOREST · R² 98% · MAE 5.98km · SCIKIT-LEARN · STREAMLIT'
    '</p>',
    unsafe_allow_html=True)