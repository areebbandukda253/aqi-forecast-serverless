import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42)
np.random.seed(42)

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
HORIZONS = [1, 2, 3]
TARGET_COLS = [f"target_day{h}" for h in HORIZONS]
LOOKBACK = 14

XGB_MAE = {"day1": 5.76, "day2": 10.37, "day3": 11.27}


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def make_sequences(X, y, lookback):
    """Turn flat rows into overlapping windows of `lookback` days."""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def build_model(n_timesteps, n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_timesteps, n_features)),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mae")
    return model


def main():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["time"])
    test = pd.read_csv(TEST_PATH, parse_dates=["time"])

    feature_cols = [c for c in train.columns if c != "time" and c not in TARGET_COLS]
    print(f"Train: {len(train)} rows | Test: {len(test)} rows | Features: {len(feature_cols)}")
    print(f"Lookback window: {LOOKBACK} days")
    print()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train[feature_cols])
    X_test_s = scaler.transform(test[feature_cols])

    results = []

    for h in HORIZONS:
        target = f"target_day{h}"
        print(f"Training day{h} ...")

        X_tr, y_tr = make_sequences(X_train_s, train[target].values, LOOKBACK)
        X_te, y_te = make_sequences(X_test_s, test[target].values, LOOKBACK)

        model = build_model(LOOKBACK, len(feature_cols))

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )

        history = model.fit(
            X_tr, y_tr,
            validation_split=0.15,
            epochs=200,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0,
            shuffle=False,
        )

        y_pred = model.predict(X_te, verbose=0).flatten()

        results.append({
            "horizon": f"day{h}",
            "epochs": len(history.history["loss"]),
            "MAE": mean_absolute_error(y_te, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_te, y_pred)),
            "R2": r2_score(y_te, y_pred),
            "MAPE": mape(y_te, y_pred),
        })
        print(f"  stopped after {len(history.history['loss'])} epochs")

    df = pd.DataFrame(results)

    print()
    print("LSTM - test set results")
    print("=" * 58)
    print(df.to_string(index=False, float_format="%.2f"))

    print()
    print("Against XGBoost")
    print("=" * 58)
    for _, row in df.iterrows():
        xgb = XGB_MAE[row["horizon"]]
        delta = xgb - row["MAE"]
        verdict = "BEATS" if delta > 0 else "loses to"
        print(f"  {row['horizon']}: {row['MAE']:.2f} vs {xgb:.2f} XGB  ->  {verdict} by {abs(delta):.2f}")

    df.to_csv("data/processed/lstm_results.csv", index=False)


if __name__ == "__main__":
    main()