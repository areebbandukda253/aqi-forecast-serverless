import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/merged_hourly.csv"
OUTPUT_PATH = "data/processed/daily_features.csv"

LAG_DAYS = [1, 2, 3, 7, 14]
ROLL_WINDOWS = [3, 7, 14, 30]
LAG_COLS = ["us_aqi_mean", "pm2_5_mean", "surface_pressure_mean"]
ROLL_COLS = ["us_aqi_mean", "pm2_5_mean"]
HORIZONS = [1, 2, 3]

DROP_COLS = ["precipitation", "dew_point_2m", "boundary_layer_height"]


def to_daily(hourly):
    """Aggregate hourly observations into one row per day."""
    hourly = hourly.copy()
    hourly["wind_dir_sin"] = np.sin(np.radians(hourly["wind_direction_10m"]))
    hourly["wind_dir_cos"] = np.cos(np.radians(hourly["wind_direction_10m"]))

    daily = hourly.set_index("time").resample("D").agg({
        "us_aqi": ["mean", "max", "min"],
        "pm2_5": "mean",
        "pm10": "mean",
        "carbon_monoxide": "mean",
        "nitrogen_dioxide": "mean",
        "sulphur_dioxide": "mean",
        "ozone": "mean",
        "temperature_2m": ["mean", "max", "min"],
        "relative_humidity_2m": "mean",
        "surface_pressure": "mean",
        "cloud_cover": "mean",
        "wind_speed_10m": ["mean", "max"],
        "wind_dir_sin": "mean",
        "wind_dir_cos": "mean",
    })
    daily.columns = ["_".join(col).strip() for col in daily.columns]
    return daily.reset_index()


def add_calendar_features(daily):
    """Cyclical encoding of the annual cycle."""
    daily = daily.copy()
    month = daily["time"].dt.month
    daily["month_sin"] = np.sin(2 * np.pi * month / 12)
    daily["month_cos"] = np.cos(2 * np.pi * month / 12)
    return daily


def add_lag_features(daily):
    """Lags, rolling statistics and change rates. All strictly backward-looking."""
    out = daily.copy().sort_values("time").reset_index(drop=True)

    for col in LAG_COLS:
        for lag in LAG_DAYS:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)

    for col in ROLL_COLS:
        for window in ROLL_WINDOWS:
            out[f"{col}_roll{window}_mean"] = out[col].rolling(window).mean()
            out[f"{col}_roll{window}_std"] = out[col].rolling(window).std()

    out["aqi_change_1d"] = out["us_aqi_mean"] - out["us_aqi_mean"].shift(1)
    out["aqi_change_3d"] = out["us_aqi_mean"] - out["us_aqi_mean"].shift(3)
    out["aqi_vs_roll7"] = out["us_aqi_mean"] - out["us_aqi_mean"].rolling(7).mean()

    return out


def add_targets(daily):
    """Future AQI values. The only place future data may appear."""
    out = daily.copy()
    for horizon in HORIZONS:
        out[f"target_day{horizon}"] = out["us_aqi_mean"].shift(-horizon)
    return out


def build(hourly, include_targets=True):
    """Full feature pipeline. Used for both training and inference."""
    daily = to_daily(hourly)
    daily = add_calendar_features(daily)
    daily = add_lag_features(daily)
    if include_targets:
        daily = add_targets(daily)
    return daily


def main():
    hourly = pd.read_csv(INPUT_PATH, parse_dates=["time"])
    hourly = hourly.drop(columns=[c for c in DROP_COLS if c in hourly.columns])
    print(f"Loaded {len(hourly)} hourly rows")

    features = build(hourly, include_targets=True)
    print(f"Built {features.shape[1]} columns for {len(features)} days")

    before = len(features)
    features = features.dropna().reset_index(drop=True)
    print(f"Dropped {before - len(features)} incomplete rows")

    features.to_csv(OUTPUT_PATH, index=False)
    print()
    print(f"Saved to {OUTPUT_PATH}")
    print(f"Shape: {features.shape}")
    print(f"Range: {features['time'].min().date()} to {features['time'].max().date()}")


if __name__ == "__main__":
    main()