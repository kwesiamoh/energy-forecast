🌍 Energy Forecasting Pipeline  
A structured, end-to-end time series forecasting pipeline for German electricity load and renewable generation, combining data engineering and machine learning.

This project builds a reproducible workflow from raw energy + weather data to multi-horizon forecasting, integrating domain-driven features with modern modeling approaches.

---

🔧 Overview  

**Phase 1 — Data Ingestion (Python)**  
- Integration of multi-source datasets: OPSD, SMARD, Meteostat  
- Hourly time-series alignment (UTC)  
- Gap handling and data validation  
- Output: unified master dataset (~80k rows, ~30 columns)  

**Phase 2 — Feature Engineering**  
- Calendar-based features (hour, weekday, seasonality, holidays)  
- Weather-derived signals (degree-days, wind power density, solar proxies)  
- Temporal features (lags, rolling statistics, differences)  
- Output: model-ready dataset (~150 features)  

**Phase 2.5 — Data Splitting & Scaling**  
- Chronological split into train / validation / test sets  
- Independent scaling to prevent data leakage  
- Configurable normalization strategies  

**Phase 3 — Baseline Models**  
- XGBoost (multi-horizon direct forecasting)  
- SARIMA (univariate statistical baseline)  
- Evaluation using MAE, RMSE, MAPE, R², SMAPE  
- Outputs: benchmark tables, plots, trained models  

**Phase 4 — Foundation Models**  (work in progress)
- Chronos (pre-trained time series foundation model)  
- Zero-shot forecasting vs fine-tuned performance  
- Comparative evaluation against classical baselines  
- Outputs: leaderboard, evaluation plots, checkpoints  

---

🎯 Purpose  
- Build a full-stack time series forecasting pipeline  
- Compare classical ML, statistical models, and foundation models  
- Explore the impact of feature engineering on energy forecasting  
- Develop reproducible, production-style data workflows  

---

⚡ Skills Demonstrated  
- Time-series data engineering and pipeline design  
- Multi-source data integration and validation  
- Feature engineering with domain knowledge (energy + weather)  
- Multi-horizon forecasting techniques  
- Model evaluation and benchmarking  
- Cross-approach modeling (XGBoost, SARIMA, foundation models)  

---

📊 Data Sources  
- Open Power System Data (OPSD) — load & generation  
- SMARD (Bundesnetzagentur) — validation overlay  
- Meteostat — weather observations (multi-station aggregation)  

---

📈 Key Outputs  
- Cleaned master dataset (`master.parquet`)  
- Feature-engineered dataset (`features.parquet`)  
- Baseline benchmark results (`baseline_table.csv`)  
- Model comparison leaderboard (`leaderboard.csv`)  
- Forecast visualizations and evaluation plots  

---

🚧 Status  
Work in progress
