# ⚡ EV Range Prediction

> Predicting the remaining range of Electric Vehicles using real-world driving, environmental, and battery features — with a Machine Learning model achieving **R² = 0.989** and **MAE = 5.89 km**.

---

## 📌 Project Overview

Electric Vehicle range anxiety is one of the biggest barriers to EV adoption. This project builds a data-driven solution to accurately predict the **remaining range (km)** of an EV based on real-time driving conditions, battery state, weather, and road characteristics.

The project covers the full data science pipeline — from **exploratory data analysis** and **visualizations** to a trained **regression model** evaluated with industry-standard metrics.

---

## 🎯 Key Results

| Metric | Score |
|---|---|
| **R² Score** | **0.9896** (98.96% variance explained) |
| **Mean Absolute Error (MAE)** | **5.89 km** |
| **Model** | Random Forest Regressor |
| **Dataset Size** | 2,000 records |

> The model explains **~99% of the variance** in EV remaining range, with an average prediction error of under 6 km.

---

## 📂 Repository Structure

```
EV-Range-Prediction/
│
├── code.py                              # Main script: EDA + ML model training & evaluation
├── sample_code.py                       # Quick sanity check / data loading test
├── EV_Range_Prediction_Dataset_2000.csv # Dataset (2000 records)
│
├── soc vs remaining range.png           # Battery SoC vs Remaining Range
├── Speed vs Power Consumption.png       # Speed vs Power Consumption (regression line)
├── Power consumption at road gradient.png
├── power consumption by driving style.png
├── Battery SOC vs Power Consumption.png
├── Driving Style vs Avg Power Consumption.png
├── Range impact by traffic density.png
├── Traffic density impact.png
├── regenerative braking vs remaining_range.png
├── Temperature vs remaining range.png
├── time series over time.png
├── Top power consumption at speed.png
├── most affect range.png                # Pairplot of key features
└── Feature Correlation Heatmap.png      # Pearson correlation heatmap
```

---

## 📊 Dataset Features

The dataset (`EV_Range_Prediction_Dataset_2000.csv`) contains **2,000 rows** with the following key columns:

| Feature | Description |
|---|---|
| `battery_soc` | Battery State of Charge (%) |
| `speed` | Vehicle speed (km/h) |
| `power_consumption` | Energy draw (kWh/km) |
| `road_gradient` | Road incline angle (%) |
| `regen_braking` | Regenerative braking intensity |
| `driving_style` | Aggressive / Normal / Eco |
| `traffic_density` | Low / Medium / High |
| `temperature` | Ambient temperature (°C) |
| `battery_health` | Battery degradation factor |
| `timestamp` | Time of record |
| `remaining_range` | **Target variable** — Remaining range (km) |

---

## 🔍 Exploratory Data Analysis

The following relationships were investigated and visualized:

- **Battery SoC vs Remaining Range** — Strong positive correlation; higher charge = more range
- **Speed vs Power Consumption** — Regression line shows power increases non-linearly with speed
- **Road Gradient Impact** — Uphill roads significantly increase energy draw
- **Driving Style Comparison** — Aggressive driving leads to higher max and average power consumption
- **Traffic Density** — Dense traffic reduces average remaining range
- **Regenerative Braking** — Higher regen values partially recover range
- **Temperature Effect** — Extreme cold/heat negatively impacts remaining range
- **Time Series** — Remaining range decay tracked over session timestamps
- **Feature Correlation Heatmap** — Identifies strongest predictors of remaining range
- **Pairplot** — Multi-feature view of `battery_soc`, `speed`, `power_consumption`, `battery_health`, `remaining_range`

---

## 🤖 Machine Learning Model

### Pipeline

```
Raw CSV → Data Cleaning (dropna) → Feature Selection → Train/Test Split → Model Training → Evaluation
```

### Steps in `code.py`

1. **Data Loading** — Read CSV with pandas
2. **Data Cleaning** — Drop rows with missing values in key columns (`battery_soc`, `power_consumption`, `speed`, `remaining_range`, `regen_braking`)
3. **Exploratory Visualization** — 13 charts covering all major feature relationships
4. **Model Training** — Random Forest Regressor trained on selected features
5. **Evaluation** — MAE and R² Score computed on test set

### Model Performance

```
Model Performance:
MAE:      5.8952
R2 Score: 0.9897
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas | Data loading & cleaning |
| Matplotlib | Custom visualizations |
| Seaborn | Statistical plots (regplot, heatmap, pairplot) |
| Scikit-learn | ML model training & evaluation |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Abhi-ops-hub/EV-Range-Prediction.git
cd EV-Range-Prediction
```

### 2. Install dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn
```

### 3. Run the main script

```bash
python code.py
```

This will:
- Load and clean the dataset
- Generate all 13 visualizations (saved as `.png` files)
- Train the Random Forest model
- Print MAE and R² Score to the console

---

## 📈 Sample Visualizations

| Chart | Insight |
|---|---|
| Feature Correlation Heatmap | `battery_soc` is the strongest predictor of remaining range |
| Speed vs Power Consumption | Power demand rises sharply above 100 km/h |
| Driving Style vs Avg Power | Aggressive style consumes ~40% more energy than Eco |
| Temperature vs Range | Range drops noticeably below 5°C and above 40°C |

---

## 💡 Key Insights

- **Battery SoC** is the single most important factor determining remaining range.
- **Driving style** has a significant behavioral impact — eco driving can extend range by up to 40%.
- **Regenerative braking** partially offsets energy loss, especially in stop-and-go traffic.
- **Temperature extremes** reduce range due to increased HVAC load and battery inefficiency.
- The ML model achieves near-perfect accuracy (**R² = 0.989**), confirming that remaining range is highly predictable from these physical features.

---

## 👤 Author

**Abhishek Goswami**
- GitHub: [@Abhi-ops-hub](https://github.com/Abhi-ops-hub)
- LinkedIn: [Abhishek Goswami](https://www.linkedin.com/in/abhishek-goswami-447185325)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
