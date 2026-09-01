import os

import hopsworks
from dotenv import load_dotenv

load_dotenv()

FG_NAME = "aqi_daily_features"
FG_VERSION = 1


def main():
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )
    fs = project.get_feature_store()

    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    print(f"Feature group: {fg.name} v{fg.version}")

    df = fg.read()

    print()
    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Range:   {df['time'].min()} to {df['time'].max()}")
    print()
    print("AQI summary:")
    print(df["us_aqi_mean"].describe().round(2))
    print()
    print("Missing values:", df.isna().sum().sum())


if __name__ == "__main__":
    main()
