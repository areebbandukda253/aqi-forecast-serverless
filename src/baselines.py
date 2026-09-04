import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
HORIZONS = [1, 2, 3]


def mape(y_true, y_pred):
    """Mean absolute percentage error."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate(y_true, y_pred, name, horizon):
    """Compute all four metrics for one set of predictions."""
    return {
        "baseline": name,
        "horizon": f"day{horizon}",
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def main():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["time"])
    test = pd.read_csv(TEST_PATH, parse_dates=["time"])

    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    print()

    results = []

    for h in HORIZONS:
        target = f"target_day{h}"
        y_true = test[target].values

        # Baseline 1: always predict the training mean
        train_mean = train[target].mean()
        y_pred = np.full(len(test), train_mean)
        results.append(evaluate(y_true, y_pred, "mean", h))

        # Baseline 2: persistence - predict today's AQI for every horizon
        y_pred = test["us_aqi_mean"].values
        results.append(evaluate(y_true, y_pred, "persistence", h))

        # Baseline 3: seasonal - predict the training average for that month
        month_avg = train.groupby(train["time"].dt.month)[target].mean()
        overall = train[target].mean()
        y_pred = test["time"].dt.month.map(month_avg).fillna(overall).values
        results.append(evaluate(y_true, y_pred, "seasonal", h))

    df = pd.DataFrame(results)

    print("Baseline results on the test set")
    print("=" * 62)
    for name in ["mean", "persistence", "seasonal"]:
        subset = df[df["baseline"] == name]
        print(f"\n{name.upper()}")
        print(subset[["horizon", "MAE", "RMSE", "R2", "MAPE"]].to_string(index=False, float_format="%.2f"))

    print()
    print("=" * 62)
    print("Best MAE per horizon (the bar to beat)")
    for h in HORIZONS:
        subset = df[df["horizon"] == f"day{h}"]
        best = subset.loc[subset["MAE"].idxmin()]
        print(f"  day{h}: {best['MAE']:.2f} MAE  ({best['baseline']})")

    df.to_csv("data/processed/baseline_results.csv", index=False)
    print()
    print("Saved to data/processed/baseline_results.csv")


if __name__ == "__main__":
    main()