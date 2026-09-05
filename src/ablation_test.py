import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

train = pd.read_csv("data/processed/train.csv", parse_dates=["time"])
test = pd.read_csv("data/processed/test.csv", parse_dates=["time"])

TARGETS = ["target_day1", "target_day2", "target_day3"]
SAME_DAY_POLLUTANTS = [
    "pm2_5_mean", "pm10_mean", "carbon_monoxide_mean",
    "nitrogen_dioxide_mean", "sulphur_dioxide_mean", "ozone_mean",
    "us_aqi_mean", "us_aqi_max", "us_aqi_min",
]

all_features = [c for c in train.columns if c != "time" and c not in TARGETS]
reduced = [c for c in all_features if c not in SAME_DAY_POLLUTANTS]

print(f"Full feature set:    {len(all_features)}")
print(f"Without same-day:    {len(reduced)}")
print()

for name, cols in [("WITH same-day", all_features), ("WITHOUT same-day", reduced)]:
    print(name)
    for h in [1, 2, 3]:
        target = f"target_day{h}"
        model = RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=4,
            max_features=0.5, random_state=42, n_jobs=-1,
        )
        model.fit(train[cols], train[target])
        pred = model.predict(test[cols])
        mae = mean_absolute_error(test[target], pred)
        r2 = r2_score(test[target], pred)
        print(f"  day{h}: MAE {mae:6.2f} | R2 {r2:5.2f}")
    print()