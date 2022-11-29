import os.path as path
import platform

import pandas as pd
import shutil

from src.domain.services import repository_service
from src.domain.utils import paths


def get_treated_dataset(overwrite_file = False):
    """Imports the treated dataset with only the columns in the project scope.

    Args:
        overwrite_file (bool, optional): If set to true, redo the process and overwrites the previously generated CSV, otherwise no process is done, just returns the existing one.

    Returns:
        DataFrame: A dataframe with the top games on Google Play.
    """
    filename = 'top_games_googleplay.csv'
    new_path = f'{paths.DATA_PATH}\\{filename}' if platform.system() == 'Windows' else f"{paths.DATA_PATH}/{filename}"

    if path.exists(new_path) and not overwrite_file:
        return pd.read_csv(new_path, sep=';')
        
    repository_service.treat_dataset(new_path)
    
    return pd.read_csv(new_path, sep=';')


def clear_temp():
    if path.exists(paths.TEMP_PATH):
        shutil.rmtree(paths.TEMP_PATH)
