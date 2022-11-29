import os
import platform

platform_system = platform.system()

DATA_PATH = f"{os.getcwd()}\\infrastructure\\data" if platform_system == 'Windows' else f"{os.getcwd()}/infrastructure/data"
"""The project relative path to the folder 'data' in infrastructure."""

DATASET_PATH = f'{DATA_PATH}\\android_apps_metadata.csv' if platform_system == 'Windows' else f'{DATA_PATH}/android_apps_metadata.csv'
"""The project relative path to the original dataset CSV file."""

TEMP_PATH = f'{DATA_PATH}\\temp' if platform_system == 'Windows' else f'{DATA_PATH}/temp'
"""The project relative path to a folder inside 'data' to store temporary files."""