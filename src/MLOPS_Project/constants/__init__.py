import os
from pathlib import Path

# Print the current working directory
# print(f"Current working directory: {os.getcwd()}")

# Define the configuration file path
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
SCHEMA_FILE_PATH = Path("schema.yaml")

# # Create directories if they don't exist
# # CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

# # # Print the paths for verification
# print(f"CONFIG_FILE_PATH: {CONFIG_FILE_PATH}")
# print(f"PARAMS_FILE_PATH: {PARAMS_FILE_PATH}")
# print(f"SCHEMA_FILE_PATH: {SCHEMA_FILE_PATH}")

# # Check if files exist
# print(f"Config file exists: {CONFIG_FILE_PATH.exists()}")
# print(f"Params file exists: {PARAMS_FILE_PATH.exists()}")
# print(f"Schema file exists: {SCHEMA_FILE_PATH.exists()}")
