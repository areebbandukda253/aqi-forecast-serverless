# Project Guide

A running journal of how this project was built — the decisions, the reasoning behind
them, and the problems encountered along the way.

Where the README says *what* the project is, this document says *why* it is that way.

---

## Table of contents

- [Phase 0 — Environment setup](#phase-0--environment-setup)
- [Phase 1 — Data collection](#phase-1--data-collection)

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

| Directory | Purpose |
|---|---|
| `data/raw/` | Exactly what the API returned. Never modified. |
| `data/processed/` | Output of cleaning and merging scripts |
| `notebooks/` | Exploratory analysis |
| `src/` | Pipeline scripts intended to run unattended |
| `models/` | Trained model artefacts |

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

| | AQICN | OpenWeather | Open-Meteo |
|---|---|---|---|
| API key required | Yes | Yes | No |
| Free historical data | Very limited | From Nov 2020 | From Aug 2022 |
| Returns AQI directly | Yes | No — raw pollutants only | Yes (`us_aqi`) |
| Weather from same provider | No | Yes | Yes |
| Daily request limit | ~1,000 | ~1,000 | 10,000 |

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

**Known limitation.** Outside Europe, Open-Meteo serves the CAMS *global* model at
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

| Variable | Physical relevance |
|---|---|
| Wind speed | Disperses pollutants. Expected to be a dominant predictor. |
| Wind direction | Determines what is carried in. Karachi is coastal: sea breeze is clean, inland flow is not. |
| Precipitation | Scavenges particulates from the air column. |
| Temperature | Drives inversions, which cap vertical mixing. |
| Relative humidity | Hygroscopic growth of particles inflates PM readings. |
| Surface pressure | High pressure implies stagnation and suppressed mixing. |
| Dew point | Combined with temperature, indicates inversion conditions. |
| Cloud cover | Modulates photochemistry, particularly ozone formation. |

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

This compares the number of hourly slots that *should* exist across the date range
against the number of rows actually present. Silent row loss during a merge — from a
timezone mismatch, a dtype mismatch, or duplicate keys — produces a file that looks
entirely normal while corrupting every downstream lag feature. The check makes that
failure loud.

### Phase 1 result

| | |
|---|---|
| Output | `data/processed/merged_hourly.csv` |
| Rows | 31,656 |
| Columns | 16 |
| Range | 2023-01-01 00:00 to present |
| Missing hours | 0 |
| Missing values | 0 |

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
