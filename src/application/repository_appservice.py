import os.path as path
import pandas as pd

from domain.services import repository_service
from domain.utils import constants


def get_treated_dataset(overwrite_file = False):
    """Imports the treated dataset with only the columns in the project scope.

    Args:
        overwrite_file (bool, optional): If set to true, redo the process and overwrites the previously generated CSV, otherwise no process is done, just returns the existing one.

    Returns:
        DataFrame: A dataframe with the top games on Google Play.
    """
    filename = 'top_games_googleplay.csv'
    new_path = f'{constants.DATA_PATH}\\{filename}'
    
    if path.exists(new_path) and not overwrite_file:
        return pd.read_csv(new_path, sep=';')
        
    repository_service.treat_dataset(new_path)
    
    return pd.read_csv(new_path, sep=';')
