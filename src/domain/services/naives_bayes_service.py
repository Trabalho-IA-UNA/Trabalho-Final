import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

from domain.models.results import Report

def naives_bayes(dataset: pd.DataFrame, target: str) -> Report: 
    
    result = Report()
    
    carac = dataset.drop([target], axis = 1)
    alvo = dataset[target]
    
    x_train,x_test,y_train,y_test = train_test_split(carac,alvo,test_size=.3, random_state=0)
    
    NB_model = GaussianNB()
    NB_model.fit(x_train,y_train)
    
    predictions = NB_model.predict(x_test)
    
    result.normalized_data = dataset
    result.score = NB_model.score(x_test, y_test)
    result.classification_report = metrics.classification_report(y_test,predictions)
    result.confusion_matrix = metrics.confusion_matrix(y_test, predictions, labels=[True, False])
    
    return result