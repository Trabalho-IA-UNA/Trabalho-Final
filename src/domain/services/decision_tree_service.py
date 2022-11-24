import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from domain.models.results import DecisionTreeResult


def decision_tree(dataset: pd.DataFrame, target: str, tree_depth: int, dataset_usage: float) -> DecisionTreeResult:
    """Executes Decision Tree on given DataFrame.

    Args:
        dataset (pd.DataFrame): The DataFrame containing the data to train on.
        target (str): The name of the target column.
        tree_depth (int): Number of tree nodes to generate from root.
        dataset_usage (float): Percentage of the dataframe (0-1) to use for training.

    Returns:
        DecisionTreeResult: A class object containig results data.
    """
    result = DecisionTreeResult()
    
    # x = training resources (predictor), y = target (predicted)
    x = dataset.drop([target],axis=1)
    y = dataset[target]
    
    x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=dataset_usage, random_state=0)

    # training and learning
    tree_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=tree_depth)
    tree_model = tree_model.fit(x_train, y_train)
    
    # predicting with the test row (x_test)
    predictions = tree_model.predict(x_test)
    
    result.tree_model = tree_model
    result.normalized_data = dataset
    result.score = tree_model.score(x_test, y_test)
    result.classification_report = classification_report(y_test, predictions)
    result.confusion_matrix = confusion_matrix(y_test, predictions, labels=[True, False])
    
    return result