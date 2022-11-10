import pandas as pd
from domain.utils import constants
import os.path as path


def import_dataset(dataset_path, separator, cols = None):
    """Imports the CSV file from the specified path.

    Args:
        dataset_path (str): The path to the file to be read.
        separator (str): The separator character of the CSV.

    Returns:
        DataFrame: A dataframe with the content of the CSV file.
    """
    
    return pd.read_csv(dataset_path, sep=separator, header=0, encoding='utf-8', usecols=cols)
