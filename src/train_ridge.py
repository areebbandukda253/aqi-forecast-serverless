import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
HORIZONS = [1, 2, 3]
TARGET_COLS = [f"target_day{h}" for h in HORIZONS]
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_feature_cols(df):
    """Every column except time and the targets."""
    return [c for c in df.columns if c != "time" and c not in TARGET_COLS]


def main():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["time"])
    test = pd.read_csv(TEST_PATH, parse_dates=["time"])

    feature_cols = get_feature_cols(train)
    print(f"Train: {len(train)} rows | Test: {len(test)} rows | Features: {len(feature_cols)}")
    print()

    X_train = train[feature_cols]
    X_test = test[feature_cols]

    results = []

    for h in HORIZONS:
        target = f"target_day{h}"
        y_train = train[target]
        y_test = test[target]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ])

        search = GridSearchCV(
            pipeline,
            param_grid={"model__alpha": ALPHAS},
            cv=TimeSeriesSplit(n_splits=5),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        best_alpha = search.best_params_["model__alpha"]
        y_pred = search.best_estimator_.predict(X_test)

        results.append({
            "horizon": f"day{h}",
            "alpha": best_alpha,
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
            "MAPE": mape(y_test.values, y_pred),
        })
        print(f"day{h}: best alpha = {best_alpha}")

    df = pd.DataFrame(results)

    print()
    print("Ridge Regression - test set results")
    print("=" * 58)
    print(df.to_string(index=False, float_format="%.2f"))

    print()
    print("Against baselines")
    print("=" * 58)
    baselines = {"day1": 9.21, "day2": 12.50, "day3": 13.88}
    for _, row in df.iterrows():
        base = baselines[row["horizon"]]
        delta = base - row["MAE"]
        verdict = "BEATS" if delta > 0 else "loses to"
        print(f"  {row['horizon']}: {row['MAE']:.2f} vs {base:.2f} baseline  ->  {verdict} by {abs(delta):.2f}")

    df.to_csv("data/processed/ridge_results.csv", index=False)


if __name__ == "__main__":
    main()