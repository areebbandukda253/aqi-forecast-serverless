import os

import hopsworks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FEATURES_PATH = "data/processed/daily_features.csv"
FG_NAME = "aqi_daily_features"
FG_VERSION = 1


def connect():
    """Log in to Hopsworks and return the feature store handle."""
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )
    print(f"Connected to project: {project.name}")
    return project.get_feature_store()


def load_features():
    """Read the engineered features and normalise column names."""
    df = pd.read_csv(FEATURES_PATH, parse_dates=["time"])
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def main():
    df = load_features()
    fs = connect()

    fg = fs.get_or_create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        description="Daily aggregated AQI and weather features for Karachi, with lags and 3-day targets",
        primary_key=["time"],
        event_time="time",
        online_enabled=False,
    )

    print(f"Uploading {len(df)} rows to {FG_NAME} v{FG_VERSION} ...")
    fg.insert(df, write_options={"wait_for_job": True})

    print()
    print("Upload complete.")
    print(f"Feature group: {fg.name} v{fg.version}")


if __name__ == "__main__":
    main()
