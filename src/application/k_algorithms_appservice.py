import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from domain.models.results import KNNResult
from domain.services import k_algorithms_service


def knn(dataset: pd.DataFrame, num_neighbors:int = 3, dataset_usage: float = 0.7) -> KNNResult:
    """Executes K-Nearest Neighbors on given DataFrame

    Args:
        dataset (pd.DataFrame): The DataFrame containing the data to train on.
        num_neighbors (int, optional): Number of neighbors to calculate distance to. Defaults to 3.
        dataset_usage (float, optional): Percentage of the dataframe (0-1) to use for training. Defaults to 0.7 (70%).

    Returns:
        KNNResult: A class object containig results data.
    """
    
    encoder = preprocessing.LabelEncoder()
    
    # converts categories into index (labels)
    categories = encoder.fit_transform(dataset["category"])
    category_column = pd.DataFrame(categories, columns=['category'])
    
    # drops text columns
    df = dataset.drop(["fancyname","category","company"],axis=1)
    
    # appending the encoded category column
    df = pd.concat([df, category_column],axis=1)
    
    # creating classification column
    df['top_rating'] = np.where(df['rating'] > 4.5, 1, 0)
    
    # dropping score columns that are not the target    
    df = df.drop(["Downloads","numberreviews","five", "four", "three", "two", "one", "rating"],axis=1)
    
    result = k_algorithms_service.knn(df, 'top_rating', num_neighbors, dataset_usage)
    
    return result
    