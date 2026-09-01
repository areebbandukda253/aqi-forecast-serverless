import os

import hopsworks
from dotenv import load_dotenv

load_dotenv()

FG_NAME = "aqi_daily_features"
FG_VERSION = 1
FV_NAME = "aqi_forecast_view"
FV_VERSION = 1

TARGET_COLS = ["target_day1", "target_day2", "target_day3"]
EXCLUDE_FROM_FEATURES = ["time"] + TARGET_COLS


def main():
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )
    fs = project.get_feature_store()

    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)

    all_cols = [f.name for f in fg.features]
    feature_cols = [c for c in all_cols if c not in EXCLUDE_FROM_FEATURES]

    print(f"Total columns:  {len(all_cols)}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Target columns:  {len(TARGET_COLS)}")

    query = fg.select_all()

    fv = fs.get_or_create_feature_view(
        name=FV_NAME,
        version=FV_VERSION,
        description="Daily AQI features with 3-day ahead targets for Karachi",
        query=query,
        labels=TARGET_COLS,
    )

    print()
    print(f"Feature view ready: {fv.name} v{fv.version}")


if __name__ == "__main__":
    main()