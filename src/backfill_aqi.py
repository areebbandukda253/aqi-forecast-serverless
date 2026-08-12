import time
from datetime import date, timedelta

import pandas as pd
import requests

AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
LATITUDE = 24.8607
LONGITUDE = 67.0011
START_DATE = date(2023, 1, 1)
CHUNK_DAYS = 90


def fetch_chunk(start, end):
    """Fetch one date range of hourly air quality data."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "auto",
    }
    response = requests.get(AQI_URL, params=params, timeout=60)
    response.raise_for_status()
    return pd.DataFrame(response.json()["hourly"])


def build_date_chunks(start, end, chunk_days):
    """Split a date range into smaller (start, end) pairs."""
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return chunks


def main():
    end_date = date.today() - timedelta(days=1)
    chunks = build_date_chunks(START_DATE, end_date, CHUNK_DAYS)
    print(f"Fetching {len(chunks)} chunks from {START_DATE} to {end_date}")

    frames = []
    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(f"  [{i}/{len(chunks)}] {chunk_start} to {chunk_end} ...", end=" ")
        chunk_df = fetch_chunk(chunk_start, chunk_end)
        print(f"{len(chunk_df)} rows")
        frames.append(chunk_df)
        time.sleep(1)

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)

    output_path = "data/raw/aqi_history.csv"
    df.to_csv(output_path, index=False)

    print()
    print(f"Saved {len(df)} rows to {output_path}")
    print(f"Range: {df['time'].min()} to {df['time'].max()}")
    print(f"Missing AQI values: {df['us_aqi'].isna().sum()}")


if __name__ == "__main__":
    main()