# 🌍 Energy and Carbon Intensity Forecasting Pipeline  

This project is an end-to-end time series forecasting pipeline for the German power grid. It predicts electricity demand and generation, and extends this by estimating the real-time carbon intensity of the grid (gCO2eq/kWh).

The idea is to combine energy system data with weather data and use machine learning to understand not just how much electricity is produced, but also how carbon-intensive it is over time.

---

## 🔧 Overview  

### ⚡ Phase 1 — Data Ingestion and Energy System Setup  
- Integration of multiple datasets: OPSD, SMARD, Meteostat  
- Full German generation mix included:
  - Lignite, Coal, Gas, Nuclear  
  - Wind, Solar, Biomass, Run-of-River, Pumped Storage  
- Separation between:
  - data ingestion (APIs / downloads)  
  - merging and backfilling logic  
- Hourly alignment (UTC), gap handling, and validation  

Derived quantities:
- Total generation  
- Residual load  
- Carbon emissions using emission factors → carbon intensity (gCO2eq/kWh)  

Output: unified dataset (~80k rows, ~30+ columns)

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

Output: model-ready dataset (~150 features)

---

### ⚙️ Phase 2.5 — Data Splitting and Scaling  
- Chronological split into train / validation / test  
- No shuffling (time order preserved)  
- Independent scaling to avoid leakage  
- Configurable normalization  

---

### 📊 Phase 3 — Baseline Models  
- XGBoost (multi-horizon forecasting)  
- SARIMA (statistical baseline)  

Forecast targets:
- Grid load (MW)  
- Carbon intensity (gCO2eq/kWh)  

Evaluation:
- MAE, RMSE, MAPE, SMAPE, R²  

Outputs:
- Benchmark tables  
- Evaluation plots  
- Trained models  

---

### 🤖 Phase 4 — Foundation Models (WIP)  
- Chronos (pre-trained time series model)  
- Comparison of:
  - zero-shot predictions  
  - fine-tuned models  

Focus:
- Generalization across time  
- Comparison with classical baselines  

Outputs:
- Model leaderboard  
- Evaluation plots  

---

## 🎯 Purpose  

- Build a full pipeline from raw data to forecasting  
- Predict both energy demand and grid carbon intensity  
- Understand how the energy mix affects emissions  
- Compare statistical models, ML models, and foundation models  
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
- Linking environmental/process engineering with ML  

---

## 📊 Data Sources  

- Open Power System Data (OPSD) — load and generation  
- SMARD (Bundesnetzagentur) — German grid data  
- Meteostat — weather data  

All data sources are open and properly attributed in the repository.

---

## 📈 Key Outputs  

- `master.parquet` — cleaned dataset  
- `features.parquet` — feature dataset  
- `baseline_table.csv` — model benchmarks  
- `leaderboard.csv` — model comparison  
- Forecast plots for load and carbon intensity  

---

## 🚧 Status  

Work in progress. Currently extending model comparison and improving evaluation.
