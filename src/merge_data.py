import pandas as pd

AQI_PATH = "data/raw/aqi_history.csv"
WEATHER_PATH = "data/raw/weather_history.csv"
OUTPUT_PATH = "data/processed/merged_hourly.csv"


def load(path):
    """Read a raw CSV and make sure 'time' is a real datetime."""
    df = pd.read_csv(path, parse_dates=["time"])
    return df.sort_values("time").reset_index(drop=True)


def main():
    aqi = load(AQI_PATH)
    weather = load(WEATHER_PATH)

    print(f"AQI:     {len(aqi):>6} rows | {aqi['time'].min()} to {aqi['time'].max()}")
    print(f"Weather: {len(weather):>6} rows | {weather['time'].min()} to {weather['time'].max()}")

    merged = pd.merge(aqi, weather, on="time", how="inner")
    merged = merged.sort_values("time").reset_index(drop=True)
    # boundary_layer_height has a 6-month gap (Jan-Jul 2024) and unreliable
    # availability at prediction time, so we exclude it entirely.
    merged = merged.drop(columns=["boundary_layer_height"])

    print(f"Merged:  {len(merged):>6} rows | {merged['time'].min()} to {merged['time'].max()}")
    print()

    expected_hours = int((merged["time"].max() - merged["time"].min()).total_seconds() / 3600) + 1
    print(f"Expected hours in range: {expected_hours}")
    print(f"Actual rows:             {len(merged)}")
    print(f"Missing hours:           {expected_hours - len(merged)}")
    print()

    print("Missing values per column:")
    missing = merged.isna().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "  None")
    print()

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")
    print(f"Shape: {merged.shape}")


if __name__ == "__main__":
    main()