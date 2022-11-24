import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

from domain.models.results import LinearRegressionResult, LogisticRegressionResult

def linear(dataset: pd.DataFrame, target:str, dataset_usage: float) -> LinearRegressionResult:
    """Executes Linear Regression on given DataFrame

    Args:
        df (pd.DataFrame): The DataFrame containing the data to train on.
        dataset_usage (float): Percentage of the dataframe (0-1) to use for training.

    Returns:
        LinearRegressionResult: A class object containig results data.
    """
    # instantiating class to store results
    result = LinearRegressionResult()
    
    dataset = (dataset-dataset.min())/(dataset.max()-dataset.min()) 
    
    # x = training resources (predictor), y = target (predicted)
    x = dataset.drop(target, axis=1, inplace=False)
    y = dataset[target]
    
    # getting train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=dataset_usage, random_state=0)
    
    # training and learning
    linear_model = LinearRegression()
    linear_model.fit(x_train,y_train)
    
    # creating dataframe to visualize each column coefficient
    coeff = pd.DataFrame(linear_model.coef_, x.columns,columns=['Coeficiente'])
    
    # predicting with the test row (x_test)
    predictions = linear_model.predict(x_test)
    
    result.linear_reg_model = linear_model
    result.normalized_data = dataset
    result.score = linear_model.score(x_test, y_test)
    result.classification_report = classification_report(y_test,predictions)
    result.confusion_matrix = confusion_matrix(y_test, predictions, labels=[True, False])
    
    result.coefficients = coeff
    result.MAE_metrics = metrics.mean_absolute_error(y_test, predictions)
    result.MSE_metrics = metrics.mean_squared_error(y_test, predictions)
    result.RMSE_metrics = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    
    result.print_overview()    
    
    return result





def logistic(dataset: pd.DataFrame, target:str, dataset_usage: float) -> LogisticRegressionResult:
    """Executes Logistic Regression on given DataFrame

    Args:
        df (pd.DataFrame): The DataFrame containing the data to train on.
        dataset_usage (float): Percentage of the dataframe (0-1) to use for training.

    Returns:
        LinearRegressionResult: A class object containig results data.
    """
    # instantiating class to store results
    result = LogisticRegressionResult()
    
    # x = training resources, y = target
    x = dataset.drop(target, axis=1, inplace=False)
    y = dataset[target]
    
    # getting train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=dataset_usage, random_state=0)
    
    # training and learning
    logistic_model = LogisticRegression(solver='lbfgs',max_iter=1000)
    logistic_model.fit(x_train,y_train)
    
    # predicting with the test row (x_test)
    predictions = logistic_model.predict(x_test)
    
    result.logistic_reg_model = logistic_model
    result.normalized_data = dataset
    result.score = logistic_model.score(x_test, y_test)
    result.classification_report = classification_report(y_test,predictions)
    result.confusion_matrix = confusion_matrix(y_test, predictions, labels=[True, False])
    
    return result
    








