# 🌍 Energy and Carbon Intensity Forecasting Pipeline  

This project is an end-to-end time series forecasting system for the German power grid. It predicts electricity demand and generation, and extends this by estimating the real-time carbon intensity of the grid (gCO2eq/kWh).

The pipeline combines energy system data with weather data and applies both classical machine learning and modern foundation models. The goal is to understand not only how much electricity is produced, but also how carbon-intensive it is, and how different modeling approaches handle this problem.

---

## 🔧 Overview  

### ⚡ Phase 1 — Data Ingestion and Energy System Setup  
- Integration of multiple datasets: OPSD, SMARD, Meteostat  
- Full German generation mix included:
  - Lignite, Coal, Gas, Nuclear  
  - Wind, Solar, Biomass, Run-of-River, Pumped Storage  
- Clean separation between ingestion and merging/backfilling logic  
- Hourly alignment (UTC), gap handling, validation  

Derived quantities:
- Total generation  
- Residual load  
- Carbon emissions → carbon intensity (gCO2eq/kWh)  

Output: unified dataset (~80k rows, ~30+ columns)

#### 📊 Macro Dataset Overview  
<img width="3634" height="1534" alt="macro_overview" src="https://github.com/user-attachments/assets/a5cd8fe9-e5f7-4b26-9f8a-665142d73bc8" />


---

### 🧠 Phase 2 — Feature Engineering  
- Calendar features (hour, weekday, seasonality, holidays)  
- Weather-based features:
  - temperature → degree-days  
  - wind → power density (∝ velocity³)  
  - solar proxies  

- Temporal features:
  - lagged values  
  - rolling statistics  
  - differences  

- Strict causal setup:
  - only past information is used  
  - prevents data leakage  

- Handling missing data:
  - removal of highly sparse sensors  
  - preservation of usable samples  

#### 🕵️‍♂️ Missing Data & Sensor Auditing  



*Auditing sensor downtime across the 8-year timeline before executing forward-filling and interpolation.*

Output: model-ready dataset (~150 features)

---

### ⚙️ Phase 2.5 — Data Splitting and Scaling  
- Chronological train / validation / test split  
- No shuffling (time order preserved)  
- Independent scaling to avoid leakage  

---

### 📊 Phase 3 — Baseline Models (XGBoost Refactor)  
- XGBoost (recursive multi-step forecasting)  
- SARIMA (statistical baseline)  

Forecast targets:
- Grid load (MW)  
- Carbon intensity (gCO2eq/kWh)  

#### 🔁 Recursive Rollout Strategy  
The XGBoost model was refactored to use a recursive rollout approach:
- trained on a single-step horizon (h = 1)  
- predictions fed back into lag features for multi-step forecasting  

This resolved the initial flat prediction issue, but revealed a key limitation:

> prediction errors compound over time, leading to drift across the 24-hour horizon  

This behavior is typical for tree-based models in autoregressive setups.

---

### 🤖 Phase 4 — Foundation Model (Chronos-T5)  

- Amazon Chronos-T5 deployed in zero-shot mode  
- Uses a 168-hour context window with self-attention  
- No manual feature engineering required  

Instead of relying on:
- weather features  
- calendar features  
- lag engineering  

the model directly learns temporal structure from raw sequences.

---

## 📊 Results  

### 168-Hour Forecast Comparison  
<img width="3634" height="1235" alt="showdown_168h" src="https://github.com/user-attachments/assets/b77a5a2d-be4a-4447-b051-90b6f1dbccba" />


Chronos tracks the actual signal much more closely over long horizons, while XGBoost gradually diverges due to recursive error accumulation.

---

### Horizon Error Propagation  
<img width="2434" height="1234" alt="horizon_error" src="https://github.com/user-attachments/assets/db70ec28-4db7-4d22-a83f-5ca9e4674888" />


- XGBoost: error increases steadily with forecast horizon  
- Chronos: remains stable across the full 24-hour window  

This highlights a structural difference:
- classical ML → error accumulation  
- foundation models → sequence-level understanding  

---

## 📈 Key Outputs  

- `master.parquet` — cleaned dataset  
- `features.parquet` — feature-engineered dataset  
- `baseline_table.csv` — model benchmarks  
- `leaderboard.csv` — model comparison  
- Forecast plots for load and carbon intensity  

### 🏆 Multi-Target R² Leaderboard  
<img width="2734" height="2689" alt="r2_leaderboard" src="https://github.com/user-attachments/assets/5b9a323d-4a77-401b-8102-c83d8873ccd1" />


---

### 🔬 Physical Insights & Uncertainty Quantification  

Beyond raw predictive accuracy, the pipeline extracts actionable grid physics and bound estimates:

#### Carbon Intensity vs. Residual Load Dependency  
<img width="2434" height="1534" alt="carbon_vs_residual_load" src="https://github.com/user-attachments/assets/fcf51b0e-f701-4c0c-9ba7-07f994ba3f9c" />

*Demonstrating highly linear grid physics: as renewables cover demand (driving residual load down), carbon intensity drops directly along the trendline.*

#### Temporal Load Shifting Opportunity  
<img width="3028" height="1384" alt="diurnal_seasonal_carbon_heatmap" src="https://github.com/user-attachments/assets/bea84602-8f3b-4497-980a-0626fdf3290f" />

*Aggregating emissions by month and hour to map exact green windows (e.g., summer midday solar abundance) for intelligent load scheduling.*

#### Foundation Model Uncertainty Quantification  
<img width="3034" height="1234" alt="chronos_uncertainty_intervals" src="https://github.com/user-attachments/assets/97f7bdb1-edd8-4441-ac09-ffdbf86f8ff1" />

*Wrapping median point forecasts in empirical 80% prediction intervals to guarantee bounded, safe estimates for grid operators.*

---

## 🎯 Purpose  

- Build a full pipeline from raw data to forecasting  
- Predict both energy demand and grid carbon intensity  
- Understand how the energy mix affects emissions  
- Compare classical models with foundation models  
- Keep everything reproducible and structured  

---

## ⚡ Skills Demonstrated  

- Time-series data processing and pipeline design  
- Integration of real-world energy and weather datasets  
- Feature engineering based on physical relationships  
- Prevention of data leakage in forecasting  
- Multi-horizon forecasting  
- Model evaluation and benchmarking  

Additional focus:
- Energy system understanding (generation mix, residual load)  
- Linking environmental/process engineering with machine learning  
- Working with foundation models for time-series forecasting  

---

## 📊 Data Sources  

- Open Power System Data (OPSD) — load and generation  
- SMARD (Bundesnetzagentur) — German grid data  
- Meteostat — weather data  

All data sources are open and properly attributed in the repository.

---

## 🧠 Conclusion  

This project shows a clear result:

> Pre-trained time-series foundation models can understand and forecast complex physical systems like power grids without task-specific feature engineering.

The Chronos model:
- outperformed a heavily engineered XGBoost pipeline  
- required no manual feature design  
- remained stable over long forecast horizons  

In contrast, classical ML models:
- depend heavily on feature engineering  
- struggle with recursive forecasting  
- accumulate errors over time  

This suggests a shift in approach:

> Instead of manually encoding domain knowledge into features, foundation models can learn these patterns directly from data.

