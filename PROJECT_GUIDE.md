# Project Guide

A running journal of how this project was built — the decisions, the reasoning behind
them, and the problems encountered along the way.

Where the README says _what_ the project is, this document says _why_ it is that way.

---

## Table of contents

- [Phase 0 — Environment setup](#phase-0--environment-setup)
- [Phase 1 — Data collection](#phase-1--data-collection)
- [Phase 2 — Exploratory data analysis](#phase-2--exploratory-data-analysis)

---

## Phase 0 — Environment setup

**Goal:** a reproducible development environment with secrets handled safely.

### Python version

Python 3.11.9, not the latest release.

Major ML libraries lag behind new Python versions by months. TensorFlow, Hopsworks, and
SHAP are all heavy dependencies that publish wheels for older interpreters first.
Building on the newest Python would surface as `No matching distribution found` errors
in later phases, with no fix except starting over.

Version 3.14 was already present on the development machine. Rather than uninstall it,
the Python launcher (`py -3.11`) was used to target 3.11 explicitly when creating the
virtual environment. Once the venv is active, `python` unambiguously means 3.11.

### Virtual environment

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

The venv isolates this project's dependencies from every other project on the machine.
It is excluded from version control — it is several hundred megabytes of libraries that
anyone can regenerate from `requirements.txt`.

### Project layout

| Directory         | Purpose                                        |
| ----------------- | ---------------------------------------------- |
| `data/raw/`       | Exactly what the API returned. Never modified. |
| `data/processed/` | Output of cleaning and merging scripts         |
| `notebooks/`      | Exploratory analysis                           |
| `src/`            | Pipeline scripts intended to run unattended    |
| `models/`         | Trained model artefacts                        |

The raw/processed separation is deliberate. If a cleaning script has a bug, the raw
files allow re-processing without re-downloading. Overwriting raw data is
unrecoverable — an API may not serve the same historical window twice.

Empty directories were given `.gitkeep` placeholder files, since Git tracks files
rather than directories and would otherwise drop them on push.

### Dependency pinning

All dependencies are pinned with `==` rather than left floating.

Unpinned dependencies resolve to whatever is newest at install time. The same
`requirements.txt` then produces different environments on different days, and code
that worked stops working for reasons unrelated to any change you made. Pinning
guarantees the local machine and the CI runner execute identical library versions.

`numpy` is held at 1.26.4 specifically — several ML libraries still break on numpy 2.x.

Libraries are added phase by phase rather than all at once. Installing the full stack
up front means long installs and version conflicts for tools that will not be touched
for weeks.

### Secrets handling

API keys are stored in a `.env` file, loaded at runtime via `python-dotenv`, and never
written into source code.

This is not a theoretical precaution. Automated bots continuously scan public GitHub
commits for strings matching known API key formats and exploit them within minutes.
The consequences are quota exhaustion, unexpected charges, and account suspension.

Two files support this:

- `.env` — real values, excluded by `.gitignore`, never leaves the machine
- `.env.example` — the same keys with empty values, committed, so collaborators know
  what is required without seeing any secret

The exclusion was verified rather than assumed:

```bash
git check-ignore -v .env
# .gitignore:151:.env    .env
```

---

## Phase 1 — Data collection

**Goal:** enough historical air quality and weather data to train a forecasting model.

### Choosing a data source

The brief suggested AQICN or OpenWeather. Both were evaluated against Open-Meteo.

|                            | AQICN        | OpenWeather              | Open-Meteo     |
| -------------------------- | ------------ | ------------------------ | -------------- |
| API key required           | Yes          | Yes                      | No             |
| Free historical data       | Very limited | From Nov 2020            | From Aug 2022  |
| Returns AQI directly       | Yes          | No — raw pollutants only | Yes (`us_aqi`) |
| Weather from same provider | No           | Yes                      | Yes            |
| Daily request limit        | ~1,000       | ~1,000                   | 10,000         |

**Open-Meteo was selected**, on three grounds:

1. **Historical depth.** The model needs years of history to learn seasonal patterns.
   A source that only serves data going forward would require months of waiting before
   training could begin — impossible on a seven-week timeline. AQICN's bulk historical
   access requires a special request with an uncertain turnaround.

2. **Single provider for both domains.** Air quality and weather come from one API with
   identical timestamp conventions, coordinate handling, and response format. This
   removes an entire category of merge bugs.

3. **No authentication.** Nothing to configure in CI, no key to expire mid-pipeline, no
   secret to leak. Anyone cloning the repository can run it immediately.

**Known limitation.** Outside Europe, Open-Meteo serves the CAMS _global_ model at
roughly 45 km resolution. These are modelled estimates, not readings from a ground
sensor in Karachi. The values are physically consistent and suitable for forecasting,
but they are not the same thing as a measurement from a specific street. A future
version could blend in ground-station readings from AQICN for validation.

An OpenWeather key will still be provisioned in a later phase as a secondary real-time
source and to exercise proper secret handling end to end.

### Understanding the target

The US AQI is a 0–500 index derived from pollutant concentrations. Each pollutant is
converted to a sub-index using EPA lookup tables, and **the reported AQI is the maximum
across all sub-indices, not the average.**

This has a visible consequence in the data: AQI frequently plateaus at a constant value
for several consecutive hours while the underlying pollutant concentrations continue to
vary. During those periods a single pollutant is dominating the index and the others are
irrelevant to it. The target variable is therefore step-like rather than smooth, which
is worth accounting for during modelling.

Scale: 0–50 good, 51–100 moderate, 101–150 unhealthy for sensitive groups,
151–200 unhealthy, 201–300 very unhealthy, 301–500 hazardous.

### Chunked backfill

A single request covering three years of hourly data is unreliable — it risks timeouts
and truncation, and a failure loses the entire download with no indication of how far it
got.

The backfill scripts therefore split the range into fixed-size windows (90 days for air
quality, 180 for weather), fetch each separately, and concatenate. A failed window can be
retried in isolation.

Three details that matter in the implementation:

- `response.raise_for_status()` — without it, an error response is passed to
  `pd.DataFrame()`, which fails later with a message that obscures the real cause.
  Failing at the point of error makes debugging tractable.
- `timeout=60` — a request without a timeout can hang indefinitely, which in a scheduled
  CI job means a run that never terminates.
- `time.sleep(1)` between requests — the API is free and imposes no such requirement,
  but hammering it is poor practice and risks throttling mid-backfill.

Data collection stops at yesterday. Today is incomplete, and partial days would
introduce a systematic distortion into any daily aggregate.

### Weather variables

Weather is not supplementary here — it drives the mechanism.

| Variable          | Physical relevance                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------- |
| Wind speed        | Disperses pollutants. Expected to be a dominant predictor.                                  |
| Wind direction    | Determines what is carried in. Karachi is coastal: sea breeze is clean, inland flow is not. |
| Precipitation     | Scavenges particulates from the air column.                                                 |
| Temperature       | Drives inversions, which cap vertical mixing.                                               |
| Relative humidity | Hygroscopic growth of particles inflates PM readings.                                       |
| Surface pressure  | High pressure implies stagnation and suppressed mixing.                                     |
| Dew point         | Combined with temperature, indicates inversion conditions.                                  |
| Cloud cover       | Modulates photochemistry, particularly ozone formation.                                     |

There is a forecasting advantage embedded here: weather is itself forecastable with
reasonable skill at a three-day horizon. Forecast weather can therefore be supplied as
input at prediction time, rather than relying solely on persistence of past pollution.

### Dropped feature: boundary layer height

Boundary layer height — the depth of the atmospheric layer into which surface pollution
mixes — was requested initially. It is physically one of the most relevant variables
available, since it effectively sets the volume of air diluting a given emission.

Inspection revealed 4,368 missing values, forming a contiguous six-month gap from
2024-01-01 to 2024-07-01. That is 13.8% of the dataset, spanning an entire winter and
spring.

Options considered:

- **Drop the affected rows.** Loses 14% of the data and, worse, punches a hole in the
  time series. Lag features such as "AQI 24 hours ago" would be undefined for every row
  following the gap.
- **Impute the gap.** Six months of a physical variable cannot be credibly synthesised.
  The model would learn from fabricated data.
- **Drop the column.** Loses one feature, retains every row.

**The column was dropped.** Beyond the arithmetic, the availability pattern is itself a
warning: a variable with unreliable historical coverage is likely to be unreliable at
inference time. A model that depends on a feature which disappears in production fails
silently and unpredictably. Several retained variables — surface pressure, temperature,
dew point — capture related information about atmospheric stability.

### Merging

Air quality and weather were joined on `time` using an inner join. Rows missing the
target cannot train; rows missing features cannot predict. Only the intersection is
useful, so non-matching rows are discarded rather than retained as nulls.

A completeness check runs after every merge:

```python
expected_hours = int((merged["time"].max() - merged["time"].min()).total_seconds() / 3600) + 1
```

This compares the number of hourly slots that _should_ exist across the date range
against the number of rows actually present. Silent row loss during a merge — from a
timezone mismatch, a dtype mismatch, or duplicate keys — produces a file that looks
entirely normal while corrupting every downstream lag feature. The check makes that
failure loud.

### Phase 1 result

|                |                                    |
| -------------- | ---------------------------------- |
| Output         | `data/processed/merged_hourly.csv` |
| Rows           | 31,656                             |
| Columns        | 16                                 |
| Range          | 2023-01-01 00:00 to present        |
| Missing hours  | 0                                  |
| Missing values | 0                                  |

### Noted for Phase 3

`wind_direction_10m` is measured in compass degrees (1–360). Supplied raw, a model
treats 359° and 1° as maximally distant when they are physically two degrees apart. The
variable must be decomposed into sine and cosine components before use:

```python
wind_dir_sin = np.sin(np.radians(wind_direction_10m))
wind_dir_cos = np.cos(np.radians(wind_direction_10m))
```

This matters more than usual for a coastal city, where wind direction distinguishes
clean marine air from polluted continental air.

## Phase 2 — Exploratory data analysis

**Goal:** understand the target well enough to design features deliberately rather than
by guesswork. Every finding below either created or eliminated a feature.

Notebook: `notebooks/01_eda.ipynb`

### The target distribution

|                     |        |
| ------------------- | ------ |
| Mean                | 90.1   |
| Median              | 82.0   |
| Min                 | 41     |
| Max                 | 297    |
| Std dev             | 27.5   |
| Interquartile range | 72–100 |

| Category                                 | Hours  | Share |
| ---------------------------------------- | ------ | ----- |
| Good (0–50)                              | 64     | 0.2%  |
| Moderate (51–100)                        | 23,806 | 75.2% |
| Unhealthy for sensitive groups (101–150) | 6,156  | 19.4% |
| Unhealthy (151–200)                      | 1,481  | 4.7%  |
| Very unhealthy (201–300)                 | 149    | 0.5%  |
| Hazardous (301+)                         | 0      | 0.0%  |

Across 3.5 years, only 64 hours registered as "Good". The observed minimum is 41 — the
air is never clean by the US standard. The distribution is right-skewed: the bulk sits
in a narrow band with a long tail toward severe episodes, which places the mean above
the median.

**Two consequences for modelling.** First, the low variance means a naive
mean-prediction achieves a deceptively low MAE, so Phase 6 must establish honest
baselines before any model is judged. Second, R² may appear modest even for a strong
model, because there is limited variance available to explain. The metric should be
interpreted against the baseline rather than against an abstract standard.

**Reporting limitation.** The observed maximum of 297 and the complete absence of
hazardous hours understate reality. Karachi does experience worse. This is the 45 km
CAMS global grid averaging local peaks across an area much larger than the city. The
system forecasts regional air quality, not street-level exposure.

### Seasonality

| Month | Mean AQI |     | Month | Mean AQI |
| ----- | -------- | --- | ----- | -------- |
| Jan   | 117.3    |     | Jul   | 84.6     |
| Feb   | 103.0    |     | Aug   | 73.5     |
| Mar   | 93.7     |     | Sep   | 75.9     |
| Apr   | 80.4     |     | Oct   | 86.9     |
| May   | 73.9     |     | Nov   | 110.9    |
| Jun   | 78.0     |     | Dec   | 106.0    |

January averages 117.3 against May's 73.9 — a 59% difference. Winter is substantially
worse than summer.

Plotting each year separately confirms the pattern is structural rather than incidental:
2023, 2024, 2025 and 2026 all trace the same U-shape, elevated at both ends of the
calendar and troughing in May–September. 2023 sits higher overall, but the shape repeats
without exception.

The mechanism is meteorological. Cooler winter air forms temperature inversions that cap
vertical mixing and trap pollution near the surface, while winds are lighter. Summer
brings monsoon activity and a stronger sea breeze off the Arabian Sea, both of which
disperse pollutants.

**Feature created:** month, encoded cyclically.

### Autocorrelation — the decisive result

| Lag                    | Autocorrelation |
| ---------------------- | --------------- |
| 1 hour                 | 0.990           |
| 6 hours                | 0.896           |
| 24 hours               | 0.754           |
| **1 day (daily mean)** | **0.835**       |
| **2 days**             | **0.677**       |
| **3 days**             | **0.590**       |
| 7 days                 | 0.459           |

This is the single most important measurement in the phase. It establishes that AQI is
forecastable at the required horizon at all.

At 0.835, yesterday's daily average explains a large share of today's. Memory decays
steadily but never vanishes — the residual 0.459 at seven days reflects the seasonal
signal persisting underneath.

**Three implications:**

1. Lag features will be the strongest predictors available. They warrant careful
   construction.
2. The decay 0.835 → 0.677 → 0.590 defines the difficulty gradient across horizons.
   Error should be expected to increase from day+1 to day+3; a flat error profile across
   horizons would indicate a bug rather than a success.
3. A persistence baseline — "tomorrow equals today" — will score well given this
   correlation. It is the bar any model must clear to justify its existence.

### Daily cycle

AQI is flat from midnight through 14:00 at approximately 89, rises to a peak of 96.5 at
19:00, then declines.

The single evening peak, rather than the twin morning-and-evening peaks characteristic
of traffic-driven pollution, indicates the dominant mechanism is the evening collapse of
the atmospheric boundary layer: as the surface cools after sunset, vertical mixing ceases
and existing pollution concentrates into a shallower volume.

The amplitude is modest — a 7-point swing on a target ranging 41–297. Real, but minor
relative to seasonal variation.

### Correlation with the target

| Variable             | Correlation with us_aqi |
| -------------------- | ----------------------- |
| pm2_5                | **0.732**               |
| sulphur_dioxide      | 0.529                   |
| carbon_monoxide      | 0.511                   |
| surface_pressure     | **0.447**               |
| dew_point_2m         | -0.429                  |
| nitrogen_dioxide     | 0.418                   |
| pm10                 | 0.361                   |
| temperature_2m       | -0.359                  |
| relative_humidity_2m | -0.295                  |
| wind_speed_10m       | -0.283                  |
| wind_direction_10m   | -0.268                  |
| ozone                | 0.265                   |
| cloud_cover          | -0.194                  |
| precipitation        | **-0.033**              |

**PM2.5 dominates at 0.732**, more than double PM10's 0.361. This identifies PM2.5 as
the pollutant most frequently setting the AQI maximum, and justifies giving it the same
lag treatment as the target itself.

**Surface pressure at 0.447** is the strongest weather variable and confirms the
stagnation hypothesis: high pressure suppresses vertical mixing and allows accumulation.
It is also the most operationally useful weather feature, because pressure is forecast
accurately several days ahead and can therefore be supplied as genuine forecast input at
inference time rather than assumed.

**An important caveat.** These are contemporaneous correlations. That PM2.5 correlates
with AQI in the same hour is close to tautological, since AQI is computed from pollutant
concentrations including PM2.5. It does not establish that today's PM2.5 predicts AQI
three days ahead. The evidence for forecastability at horizon comes from the
autocorrelation figures, not from this table.

### Features eliminated

Four candidate features were removed on evidence rather than convenience. Each had a
plausible prior justification.

| Feature                 | Reason                                                                                                                                                                                                                                                                                                   |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `precipitation`         | Correlation -0.033. Karachi is arid: the 75th percentile of hourly rainfall is 0.0, so the variable is zero for the overwhelming majority of observations and offers almost nothing to learn from. The physical mechanism — wet deposition scavenging particulates — is genuine, but rarely active here. |
| `dayofweek`             | Mean AQI is 88–91 across all seven days, with no discernible weekday/weekend structure. The expected commuter-traffic signature is absent, either because traffic is not the dominant emission source or because the 45 km grid averages it away.                                                        |
| `dew_point_2m`          | Correlates 0.80 with relative humidity and -0.77 with surface pressure. Dew point is derived from temperature and humidity and therefore contributes no independent information, while diluting feature importance across collinear variables and obscuring SHAP interpretation.                         |
| `boundary_layer_height` | Removed in Phase 1: contiguous six-month gap and unreliable availability at inference time.                                                                                                                                                                                                              |

Two of these — precipitation and day-of-week — were predicted to be important on
physical and behavioural grounds before the data was examined. Both were wrong. This is
the principal justification for conducting EDA before feature engineering rather than
after.

### Multicollinearity noted

Several predictor pairs are strongly related and should be watched during feature
selection and SHAP interpretation:

| Pair                               | Correlation |
| ---------------------------------- | ----------- |
| dew_point ↔ relative_humidity      | 0.80        |
| carbon_monoxide ↔ nitrogen_dioxide | 0.79        |
| dew_point ↔ surface_pressure       | -0.77       |
| temperature ↔ surface_pressure     | -0.69       |

CO and NO2 at 0.79 share a common source in combustion. Both are retained for now, since
tree-based models tolerate collinearity for prediction, but their importance scores
should be read as a pair rather than independently.

### Resulting feature plan for Phase 3

| Status                     | Variables                                                                                                                            |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Core, full lag treatment   | `us_aqi`, `pm2_5`                                                                                                                    |
| Strong predictors          | `surface_pressure`, `sulphur_dioxide`, `carbon_monoxide`                                                                             |
| Retained                   | `pm10`, `nitrogen_dioxide`, `ozone`, `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`, `cloud_cover` |
| Cyclical encoding required | `month`, `wind_direction_10m`                                                                                                        |
| Eliminated                 | `precipitation`, `dew_point_2m`, `dayofweek`, `boundary_layer_height`                                                                |
