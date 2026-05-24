# 🌍 Energy and Carbon Intensity Forecasting Pipeline

This project builds an end-to-end forecasting pipeline for the German power grid.

The work focuses on two connected questions:

1. How much electricity will the grid need?
2. How carbon-intensive will that electricity be?

Instead of treating electricity demand as a purely numerical forecasting task, this project links demand, generation mix, weather conditions, and emissions into one structured pipeline. The aim is to forecast not only grid load, but also the carbon intensity of electricity in real time, measured in gCO₂eq/kWh.

This makes the problem more practical: electricity is not equally clean at every hour of the day. A kilowatt-hour consumed during high renewable generation can have a very different emissions impact from one consumed during fossil-heavy periods.

## TL;DR

This project builds an end-to-end forecasting pipeline for the German power grid, predicting both electricity demand and real-time carbon intensity.

It combines open grid data, weather data, physical feature engineering, classical machine learning, and a time-series foundation model to study how well different approaches forecast energy system behavior.

The pipeline estimates residual load, emissions, and carbon intensity, then compares an engineered XGBoost forecasting setup against Amazon Chronos-T5 in zero-shot mode.

The key finding is that XGBoost performs well at short horizons but accumulates error during recursive multi-step forecasting, while Chronos remains more stable over longer forecast windows.

Beyond prediction accuracy, the project also identifies cleaner electricity windows and shows how carbon intensity forecasting can support more emissions-aware energy scheduling.

---

## 🔧 Overview

The pipeline combines German power system data, weather data, physical feature engineering, classical machine learning, and time-series foundation models.

It follows the full workflow from raw data ingestion to model evaluation:

- collecting and aligning grid and weather data
- estimating emissions and carbon intensity
- engineering physically meaningful forecasting features
- training baseline forecasting models
- comparing classical machine learning with a pre-trained foundation model
- analyzing forecast stability, error propagation, and grid-level carbon patterns

---

## ⚡ Phase 1 — Data Ingestion and Energy System Setup

The first step was to build a reliable hourly dataset for the German power grid.

I combined multiple open datasets, including OPSD, SMARD, and Meteostat, and aligned them into a common hourly UTC timeline.

The dataset includes the main components of the German generation mix:

- lignite
- hard coal
- gas
- nuclear
- wind
- solar
- biomass
- run-of-river hydro
- pumped storage

From these raw inputs, the pipeline derives several quantities that are important for both forecasting and interpretation:

- total electricity generation
- residual load
- estimated carbon emissions
- carbon intensity in gCO₂eq/kWh

The result is a unified dataset with roughly 80,000 hourly observations and more than 30 core variables.

#### 📊 Macro Dataset Overview

<img width="3634" height="1534" alt="macro_overview" src="https://github.com/user-attachments/assets/a5cd8fe9-e5f7-4b26-9f8a-665142d73bc8" />

---

## 🧠 Phase 2 — Feature Engineering

After building the core dataset, I developed a feature engineering layer for time-series forecasting.

The features are designed to capture three types of structure:

### Calendar structure

Electricity demand follows strong daily, weekly, and seasonal rhythms, so the pipeline includes:

- hour of day
- weekday
- weekend indicators
- seasonal patterns
- holiday effects

### Weather-driven structure

Weather affects both electricity demand and renewable generation. The pipeline includes features such as:

- temperature-based heating and cooling degree indicators
- wind power density proxies based on wind speed
- solar-related proxies
- weather station aggregation and validation

### Temporal structure

To help the models learn from recent system behavior, the pipeline adds:

- lagged values
- rolling statistics
- differences
- historical target behavior

A strict causal setup is used throughout. The model only receives information that would have been available at prediction time. This avoids data leakage and keeps the forecasting problem realistic.

#### 🕵️‍♂️ Missing Data & Sensor Auditing

<img width="3034" height="1534" alt="missing_data_heatmap" src="https://github.com/user-attachments/assets/218cd5ae-ac92-43ef-bff3-70a7f985154b" />

*Sensor availability was audited across the full 8-year timeline before applying forward-filling, interpolation, and sparse-sensor filtering.*

The final feature-engineered dataset contains around 150 model-ready features.

---

## ⚙️ Phase 2.5 — Data Splitting and Scaling

The data is split chronologically into training, validation, and test sets.

No shuffling is used, because time order matters in forecasting. The model is evaluated on future periods that were not seen during training.

Scaling is also handled carefully to prevent leakage between training and evaluation periods.

---

## 📊 Phase 3 — Baseline Models

The first modeling stage uses classical forecasting baselines.

The main machine learning baseline is XGBoost, implemented with a recursive multi-step forecasting strategy. A SARIMA model is also included as a statistical reference point.

The forecast targets are:

- grid load in MW
- carbon intensity in gCO₂eq/kWh

---

## 🔁 Recursive Rollout Strategy

The XGBoost model is trained for a one-step-ahead forecast.

To predict multiple hours into the future, the model feeds its own prediction back into the lag features and repeats the process across the forecast horizon.

This solved the initial issue of overly flat predictions, but it also exposed a known limitation of recursive forecasting:

> small errors at early horizons become inputs for later horizons, causing forecast drift over time.

This behavior is especially visible across a 24-hour window. XGBoost can track short-term changes, but its stability decreases as the horizon grows.

---

## 🤖 Phase 4 — Foundation Model: Chronos-T5

To compare the classical approach with a modern sequence model, I tested Amazon Chronos-T5.

Chronos was used in zero-shot mode. It was not fine-tuned on this specific German grid dataset.

Instead of relying on manually engineered features, Chronos receives a 168-hour context window and forecasts directly from the raw time-series sequence.

This creates a useful comparison between two modeling philosophies:

- XGBoost depends on carefully designed domain features.
- Chronos relies on pre-trained temporal sequence understanding.

The question is whether a foundation model can capture the structure of a physical energy system without requiring hand-built weather, calendar, and lag features.

---

## 📊 Results

### 168-Hour Forecast Comparison

<img width="3634" height="1235" alt="showdown_168h" src="https://github.com/user-attachments/assets/b77a5a2d-be4a-4447-b051-90b6f1dbccba" />

Chronos follows the actual signal more closely across longer horizons.

XGBoost performs reasonably in the near term, but gradually moves away from the observed trajectory as recursive errors accumulate.

This difference becomes important when the forecast is used for planning rather than only short-term prediction.

---

### Horizon Error Propagation

<img width="2434" height="1234" alt="horizon_error" src="https://github.com/user-attachments/assets/db70ec28-4db7-4d22-a83f-5ca9e4674888" />

The horizon-level error analysis shows a clear pattern:

- XGBoost error increases as the forecast horizon grows.
- Chronos remains more stable across the full 24-hour window.

This points to a structural difference between the approaches. Recursive tree-based models are sensitive to their own previous mistakes, while the foundation model forecasts the sequence more directly.

---

## 📈 Key Outputs

The pipeline produces the following main artifacts:

- `master.parquet` — cleaned and merged hourly dataset
- `features.parquet` — feature-engineered forecasting dataset
- `baseline_table.csv` — baseline model metrics
- `leaderboard.csv` — model comparison table
- forecast plots for load and carbon intensity
- residual load and carbon intensity analysis
- uncertainty interval visualizations

### 🏆 Multi-Target R² Leaderboard

<img width="2734" height="2689" alt="r2_leaderboard" src="https://github.com/user-attachments/assets/5b9a323d-4a77-401b-8102-c83d8873ccd1" />

---

## 🔬 Physical Insights

The project is not only about model accuracy. It also uses the data and forecasts to study how the grid behaves physically.

### Carbon Intensity and Residual Load

<img width="2434" height="1534" alt="carbon_vs_residual_load" src="https://github.com/user-attachments/assets/fcf51b0e-f701-4c0c-9ba7-07f994ba3f9c" />

Residual load is the part of demand that remains after renewable generation has been accounted for.

When residual load is low, renewables are covering more of the system demand. In those periods, carbon intensity tends to drop.

The relationship is strongly visible in the data: as residual load decreases, carbon intensity decreases as well. This gives the model output a physical interpretation, not just a statistical one.

---

### Temporal Load Shifting Opportunity

<img width="3028" height="1384" alt="diurnal_seasonal_carbon_heatmap" src="https://github.com/user-attachments/assets/bea84602-8f3b-4497-980a-0626fdf3290f" />

Carbon intensity changes by both hour of day and season.

The heatmap highlights cleaner electricity windows, such as periods with strong solar generation during summer midday hours.

This is useful for carbon-aware scheduling. Flexible electricity use, such as storage charging or energy-intensive processes, can be shifted toward lower-carbon periods when operationally possible.

---

### Foundation Model Uncertainty Quantification

<img width="3034" height="1234" alt="chronos_uncertainty_intervals" src="https://github.com/user-attachments/assets/97f7bdb1-edd8-4441-ac09-ffdbf86f8ff1" />

The Chronos forecasts are extended with empirical 80% prediction intervals.

This gives a range of plausible future values instead of only a single forecast line.

For grid-related applications, this matters because uncertainty affects planning. A forecast is more useful when it also communicates how confident the model is across the prediction horizon.

---

## 🎯 Purpose

This project was designed to connect machine learning with real energy-system behavior.

The main objectives are to:

- build a reproducible forecasting pipeline from raw data to evaluation
- forecast both electricity demand and grid carbon intensity
- study how the generation mix affects emissions
- compare classical machine learning with time-series foundation models
- keep the modeling setup causal and leakage-free
- extract interpretable physical insights from the results

---

## ⚡ Skills Demonstrated

This project demonstrates work across data engineering, forecasting, machine learning, and energy-system analysis.

Main technical areas:

- time-series data processing
- multi-source dataset integration
- weather and power system data alignment
- physical feature engineering
- leakage-safe forecasting design
- recursive multi-step forecasting
- model benchmarking and evaluation
- foundation model testing for time-series data
- uncertainty estimation
- carbon intensity analysis

Energy and environmental focus:

- German generation mix modeling
- residual load analysis
- emissions estimation
- carbon-aware electricity use
- linking process/environmental engineering with machine learning

---

## 📊 Data Sources

The pipeline uses open data sources:

- Open Power System Data — load and generation data
- SMARD / Bundesnetzagentur — German electricity market and grid data
- Meteostat — historical weather data

All sources are open and attributed in the repository.

---

## 🧠 Conclusion

The results show that carbon intensity forecasting can be approached as both a machine learning problem and an energy-systems problem.

The engineered XGBoost pipeline provides a strong classical baseline, but its recursive setup leads to increasing error across longer forecast horizons.

Chronos-T5 performs more stably over extended horizons, even without manual feature engineering or fine-tuning on the specific dataset.

This suggests that pre-trained time-series foundation models can capture meaningful structure in complex physical systems such as power grids.

The practical value is clear: better carbon intensity forecasts can support electricity use that is not only demand-aware, but emissions-aware.

By identifying when electricity is likely to be cleaner, this type of pipeline can help guide smarter scheduling, flexible demand, and lower-carbon operation of energy systems.
