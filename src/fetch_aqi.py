import requests

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

params = {
    "latitude": 24.8607,
    "longitude": 67.0011,
    "hourly": "pm10,pm2_5,us_aqi",
    "timezone": "auto",
    "forecast_days": 1,
}

response = requests.get(URL, params=params)

print("Status code:", response.status_code)
print("Actual URL called:", response.url)

data = response.json()

print("Top-level keys:", list(data.keys()))
print("Hourly keys:", list(data["hourly"].keys()))
print("Number of timestamps:", len(data["hourly"]["time"]))
print("First 3 times:", data["hourly"]["time"][:3])
print("First 3 AQI values:", data["hourly"]["us_aqi"][:3])

#data frame
import pandas as pd

df = pd.DataFrame(data["hourly"])

print()
print("Shape:", df.shape)
print(df.head())
print()
print(df.dtypes)

df["time"] = pd.to_datetime(df["time"])

print()
print("Fixed dtype:", df["time"].dtype)
print("Hour of first row:", df["time"].dt.hour.iloc[0])