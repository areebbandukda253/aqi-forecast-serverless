import os
from dotenv import load_dotenv

load_dotenv()

city = os.getenv("CITY_NAME")
lat = os.getenv("LATITUDE")
lon = os.getenv("LONGITUDE")
key = os.getenv("OPENWEATHER_API_KEY")

print("City:", city)
print("Coordinates:", lat, lon)
print("API key found:", key is not None)