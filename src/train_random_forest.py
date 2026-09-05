import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
HORIZONS = [1, 2, 3]
TARGET_COLS = [f"target_day{h}" for h in HORIZONS]

PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6, 8, 12, None],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": [0.3, 0.5, 0.7, 1.0],
}

RIDGE_MAE = {"day1": 6.51, "day2": 10.77, "day3": 11.57}


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_feature_cols(df):
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
    importances = {}

    for h in HORIZONS:
        target = f"target_day{h}"
        print(f"Tuning day{h} ...")

        search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_distributions=PARAM_GRID,
            n_iter=25,
            cv=TimeSeriesSplit(n_splits=5),
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, train[target])

        best = search.best_estimator_
        y_pred = best.predict(X_test)

        results.append({
            "horizon": f"day{h}",
            "MAE": mean_absolute_error(test[target], y_pred),
            "RMSE": np.sqrt(mean_squared_error(test[target], y_pred)),
            "R2": r2_score(test[target], y_pred),
            "MAPE": mape(test[target].values, y_pred),
        })

        importances[f"day{h}"] = pd.Series(
            best.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)

        print(f"  best params: {search.best_params_}")

    df = pd.DataFrame(results)

    print()
    print("Random Forest - test set results")
    print("=" * 58)
    print(df.to_string(index=False, float_format="%.2f"))

    print()
    print("Against Ridge")
    print("=" * 58)
    for _, row in df.iterrows():
        ridge = RIDGE_MAE[row["horizon"]]
        delta = ridge - row["MAE"]
        verdict = "BEATS" if delta > 0 else "loses to"
        print(f"  {row['horizon']}: {row['MAE']:.2f} vs {ridge:.2f} Ridge  ->  {verdict} by {abs(delta):.2f}")

    print()
    print("Top 10 features for day1")
    print("=" * 58)
    print(importances["day1"].head(10).to_string(float_format="%.4f"))

    df.to_csv("data/processed/rf_results.csv", index=False)


if __name__ == "__main__":
    main()