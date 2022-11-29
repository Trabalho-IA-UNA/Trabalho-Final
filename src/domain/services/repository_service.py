import os.path as path
import pandas as pd
import numpy as np

from src.infrastructure import repository
from src.domain.utils import constants
from src.domain.utils import paths


def import_dataset(cols=None):
    """Imports the CSV file from the configuration path.

    Args:
        cols (list, optional): A list of the columns to read. Defaults to None will read all.

    Returns:
        DataFrame: A dataframe with the content of the file, or empty if not found.
    """

    if path.exists(paths.DATASET_PATH):
        return repository.import_dataset(paths.DATASET_PATH, constants.DATASET_SEPARATOR, cols)
    return pd.DataFrame()


def treat_dataset(new_path):
    """Treats the data, removing invalid values, fixing data types and filtering for Google Play games. Saves to a new CSV.

    Args:
        new_path (str): The path to write the treated CSV.
    """

    # import dataset with only the desired columns plus game and source for filtering
    df = import_dataset(constants.SCOPE_COLUMNS + ['game', 'source'])

    # filtering for games only and 'google play' source only
    df = df[(df['game'] == 1) and (df['source'] == 'google play')]

    # dropping columns used only for filtering
    df.drop(['game', 'source'], axis=1, inplace=True)

    error_values = ['no info available', 'error during scraping', 'no info', 'rating disabled']

    # removing rows with errors
    for col in df.columns:
        df = not df.loc[df[col].isin(error_values)]

    def remove_plus(item):
        return str(item).replace('+', '')

    def value_or_zero(item, default_str):
        return 0 if str(item) == default_str else item

    # mapping age rating to remove '+' and replace 'everyone' for 0
    df['age_rating'] = df['age_rating'].apply(lambda item: remove_plus(value_or_zero(item, 'everyone')))
    # mapping Downloads to remove '+'
    df['Downloads'] = df['Downloads'].apply(lambda item: int(remove_plus(item).replace(',', '')))
    # mapping price to replace 'free' for 0
    df['price'] = df['price'].apply(lambda item: value_or_zero(item, 'free'))
    # mapping reviews to int
    df['numberreviews'] = df['numberreviews'].apply(lambda item: int(str(item).replace(',', '')))

    # column names of start reviews
    star_nums = ["five", "four", "three", "two", "one"]
    stars = df[star_nums]

    def get_percentage(row):
        _row = row.astype(np.float16)
        for i in range(len(_row)):
            _row[i] /= sum(_row)

        return _row

    # converts relative proportion to percentage
    stars = stars.apply(get_percentage, axis=1)

    # #removes old star reviews and add the new ones
    df.drop(star_nums, axis=1, inplace=True)
    df = pd.concat([df, stars], axis=1)

    # saves the formatted version to a CSV
    df.to_csv(new_path, index=False, sep=';')
