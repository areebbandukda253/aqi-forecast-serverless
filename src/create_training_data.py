import os

import hopsworks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FV_NAME = "aqi_forecast_view"
FV_VERSION = 1
TEST_FRACTION = 0.2

OUT_DIR = "data/processed"


def main():
    project = hopsworks.login(
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_daily_features", version=1)

    print("Reading feature group ...")
    df = fg.read()
    df = df.sort_values("time").reset_index(drop=True)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    split_idx = int(len(df) * (1 - TEST_FRACTION))
    split_date = df.loc[split_idx, "time"]

    train = df[df["time"] < split_date].reset_index(drop=True)
    test = df[df["time"] >= split_date].reset_index(drop=True)

    print()
    print(f"Split date: {split_date.date()}")
    print(f"Train: {len(train):>5} rows | {train['time'].min().date()} to {train['time'].max().date()}")
    print(f"Test:  {len(test):>5} rows | {test['time'].min().date()} to {test['time'].max().date()}")

    assert train["time"].max() < test["time"].min(), "Train and test overlap in time"
    print()
    print("No temporal overlap between train and test.")

    train.to_csv(f"{OUT_DIR}/train.csv", index=False)
    test.to_csv(f"{OUT_DIR}/test.csv", index=False)
    print(f"Saved to {OUT_DIR}/train.csv and {OUT_DIR}/test.csv")

    print()
    print("Target summary:")
    for col in ["target_day1", "target_day2", "target_day3"]:
        print(f"  {col}: train mean {train[col].mean():6.2f} | test mean {test[col].mean():6.2f}")


if __name__ == "__main__":
    main()