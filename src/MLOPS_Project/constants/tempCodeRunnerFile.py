# Create directories if they don't exist
CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Print the paths for verification
print(f"CONFIG_FILE_PATH: {CONFIG_FILE_PATH}")
print(f"PARAMS_FILE_PATH: {PARAMS_FILE_PATH}")
print(f"SCHEMA_FILE_PATH: {SCHEMA_FILE_PATH}")

# Check if files exist
print(f"Config file exists: {CONFIG_FILE_PATH.exists()}")
print(f"Params file exists: {PARAMS_FILE_PATH.exists()}")
print(f"Schema file exists: {SCHEMA_FILE_PATH.exists()}")