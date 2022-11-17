import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from domain.models.results import KNNResult



def knn(dataset: pd.DataFrame, target: str, num_neighbors:int, dataset_usage: float) -> KNNResult:
    """Executes K-Nearest Neighbors on given DataFrame.

    Args:
        dataset (pd.DataFrame): The DataFrame containing the data to train on.
        target (str): The name of the target column.
        num_neighbors (int): Number of neighbors to calculate distance to.
        dataset_usage (float): Percentage of the dataframe (0-1) to use for training.

    Returns:
        KNNResult: A class object containig results data.
    """
    
    result = KNNResult()
    
    # x = training resources (predictor), y = target (predicted)
    x = dataset.drop([target],axis=1).values
    y = dataset[target].values
    
    # getting train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=dataset_usage, random_state=0, stratify=y)
    
    # training and learning
    knn_model = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn_model.fit(x_train, y_train)
            
    # predicting with the test row (x_test)
    predictions = knn_model.predict(x_test)
    
    result.knn_model = knn_model
    result.normalized_data = dataset
    result.score = knn_model.score(x_test, y_test)
    result.classification_report = classification_report(y_test, predictions)
    result.confusion_matrix = confusion_matrix(y_test, predictions, labels=[True, False])
    
    return result