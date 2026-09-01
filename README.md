# 3-Day City-Level AQI Forecasting

A fully serverless machine learning system that forecasts the Air Quality Index (AQI)
for Karachi, Pakistan, up to three days ahead.

Built during the 10Pearls Shine Internship (Cohort 9, Data Science).

> **Status:** In development. Phase 1 of 20 complete.

---

## Overview

Air pollution is a daily health concern in Karachi, where AQI regularly exceeds the
"unhealthy for sensitive groups" threshold. This project predicts AQI for the next
three days so that residents can plan ahead.

The system runs entirely on free, serverless infrastructure. There is no server to
provision, pay for, or maintain — scheduled jobs wake up, do their work, and stop.

## Architecture

```
Weather & Air Quality APIs
          |
          v
   Feature pipeline  (hourly, GitHub Actions)
          |
          v
    Feature Store  (Hopsworks)
          |
          +-----------------------------+
          |                             |
          v                             |
   Training pipeline  (daily)           |
          |                             |
          v                             v
    Model Registry  --------->  Streamlit dashboard
```

The feature store sits at the centre of the design. Both training and inference read
their features from the same place, which prevents training-serving skew — the common
failure where a model is trained on one version of the data and served a subtly
different one.

## Tech stack

| Layer          | Technology                                                    |
| -------------- | ------------------------------------------------------------- |
| Language       | Python 3.11                                                   |
| Data           | Pandas, NumPy                                                 |
| Data sources   | Open-Meteo Air Quality API, Open-Meteo Historical Weather API |
| Feature store  | Hopsworks                                                     |
| Modelling      | Scikit-learn, XGBoost, TensorFlow                             |
| Explainability | SHAP                                                          |
| Model registry | Hopsworks                                                     |
| API            | FastAPI                                                       |
| Dashboard      | Streamlit                                                     |
| Orchestration  | GitHub Actions                                                |

## Data

|           |                                          |
| --------- | ---------------------------------------- |
| Location  | Karachi, Pakistan (24.8607 N, 67.0011 E) |
| Period    | January 2023 – present                   |
| Frequency | Hourly                                   |
| Records   | ~31,600                                  |
| Target    | US AQI (0–500 scale)                     |

**Pollutants:** PM2.5, PM10, carbon monoxide, nitrogen dioxide, sulphur dioxide, ozone

**Weather:** temperature, relative humidity, dew point, precipitation, surface pressure,
cloud cover, wind speed, wind direction

Air quality data comes from the Copernicus Atmosphere Monitoring Service (CAMS) via
Open-Meteo. Historical weather comes from ECMWF ERA5 reanalysis via the same provider.

## Getting started

**Prerequisites:** Python 3.11, Git

```bash
git clone https://github.com/areebbandukda253/aqi-forecast-serverless.git
cd aqi-forecast-serverless

python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS / Linux

pip install -r requirements.txt
cp .env.example .env
```

No API key is required for data collection — Open-Meteo serves open data without
authentication.

## Usage

```bash
python src/backfill_aqi.py       # download historical air quality
python src/backfill_weather.py   # download historical weather
python src/merge_data.py         # join into a single hourly dataset
```

## Project structure

```
aqi-forecast-serverless/
├── data/
│   ├── raw/              # untouched API output
│   └── processed/        # cleaned and merged datasets
├── notebooks/            # exploratory analysis
├── src/                  # pipeline scripts
├── models/               # trained model artefacts
├── .env.example          # required environment variables
├── requirements.txt
├── README.md
└── PROJECT_GUIDE.md      # detailed development journal
```

## Roadmap

- [x] Phase 0 — Environment and repository setup
- [x] Phase 1 — Historical data collection
- [x] Phase 2 — Data cleaning and exploratory analysis
- [x] Phase 3 — Feature engineering
- [x] Phase 4 — Feature store
- [x] Phase 5 — Training dataset creation
- [ ] Phase 6 — Baseline model
- [ ] Phase 7 — Model experiments
- [ ] Phase 8 — Evaluation and selection
- [ ] Phase 9 — Model registry
- [ ] Phase 10 — 3-day prediction pipeline
- [ ] Phase 11 — Prediction API
- [ ] Phase 12 — Streamlit dashboard
- [ ] Phase 13 — Serverless deployment
- [ ] Phase 14 — Automated feature pipeline
- [ ] Phase 15 — Automated training
- [ ] Phase 16 — CI/CD orchestration
- [ ] Phase 17 — Monitoring and logging
- [ ] Phase 18 — AQI alerts
- [ ] Phase 19 — Final testing
- [ ] Phase 20 — Documentation

## Attribution

Weather and air quality data by [Open-Meteo.com](https://open-meteo.com), licensed
under CC BY 4.0. Air quality forecasts are produced by the Copernicus Atmosphere
Monitoring Service (CAMS). Historical weather is derived from ECMWF ERA5 reanalysis.
