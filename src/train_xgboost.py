import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
HORIZONS = [1, 2, 3]
TARGET_COLS = [f"target_day{h}" for h in HORIZONS]

PARAM_GRID = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4, 6],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.3, 0.5, 0.8],
    "reg_lambda": [1.0, 5.0, 20.0],
    "min_child_weight": [1, 5, 10],
}

RF_MAE = {"day1": 5.85, "day2": 10.43, "day3": 11.27}


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["time"])
    test = pd.read_csv(TEST_PATH, parse_dates=["time"])

    feature_cols = [c for c in train.columns if c != "time" and c not in TARGET_COLS]
    print(f"Train: {len(train)} rows | Test: {len(test)} rows | Features: {len(feature_cols)}")
    print()

    X_train, X_test = train[feature_cols], test[feature_cols]
    results = []
    importances = {}

    for h in HORIZONS:
        target = f"target_day{h}"
        print(f"Tuning day{h} ...")

        search = RandomizedSearchCV(
            XGBRegressor(random_state=42, n_jobs=-1, objective="reg:absoluteerror"),
            param_distributions=PARAM_GRID,
            n_iter=40,
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
    print("XGBoost - test set results")
    print("=" * 58)
    print(df.to_string(index=False, float_format="%.2f"))

    print()
    print("Against Random Forest")
    print("=" * 58)
    for _, row in df.iterrows():
        rf = RF_MAE[row["horizon"]]
        delta = rf - row["MAE"]
        verdict = "BEATS" if delta > 0 else "loses to"
        print(f"  {row['horizon']}: {row['MAE']:.2f} vs {rf:.2f} RF  ->  {verdict} by {abs(delta):.2f}")

    print()
    print("Top 10 features for day3")
    print("=" * 58)
    print(importances["day3"].head(10).to_string(float_format="%.4f"))

    df.to_csv("data/processed/xgb_results.csv", index=False)


if __name__ == "__main__":
    main()