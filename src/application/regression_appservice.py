import pandas as pd

from domain.models.results import LinearRegressionResult
from domain.services import regression_service

def linear(dataset: pd.DataFrame, dataset_usage: float = 1) -> LinearRegressionResult:
    """Executes Linear Regression on given DataFrame

    Args:
        dataset (pd.DataFrame): The DataFrame containing the data to train on.
        dataset_usage (float, optional): Percentage of the dataframe (0-1) to use for training. Defaults to 1 (100%).

    Returns:
        LinearRegressionResult: A class object containig results data.
    """
    
    # converts categories into columns (dummies)
    categories = pd.get_dummies(dataset["category"],drop_first=True)
    categories = pd.DataFrame.add_prefix(categories, 'category_')

    # drops text columns
    dataset.drop(["fancyname","category","company"],axis=1,inplace=True)
    
    # drops score columns that are not the target
    dataset.drop(["Downloads","numberreviews","five", "four", "three", "two", "one"],axis=1,inplace=True)
    
    # adds categories as dummies
    df = pd.concat([dataset,categories],axis=1)
    
    result = regression_service.linear(df, "rating", dataset_usage)
    
    return result
    








