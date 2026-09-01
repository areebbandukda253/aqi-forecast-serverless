import os

import hopsworks
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT")

if not api_key or api_key == "not_yet":
    raise SystemExit("HOPSWORKS_API_KEY is missing from .env")

print(f"Connecting to project: {project_name}")

project = hopsworks.login(
    project=project_name,
    api_key_value=api_key,
)

print()
print(f"Connected as:   {project.owner}")
print(f"Project name:   {project.name}")
print(f"Project ID:     {project.id}")

fs = project.get_feature_store()
print(f"Feature store:  {fs.name}")

feature_groups = fs.get_feature_groups()
print(f"Feature groups: {len(feature_groups)}")
