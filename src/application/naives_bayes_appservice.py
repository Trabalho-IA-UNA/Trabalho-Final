import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from domain.services import naives_bayes_service


from domain.services import repository_service
from domain.utils import constants

def naives_bayes(dataset: pd.DataFrame):
    
    encoder = preprocessing.LabelEncoder()
    categoria_encoded = encoder.fit_transform(dataset["category"])
    
    dataset['category'] = categoria_encoded
    
    # drops text columns
    df = dataset.drop(["fancyname","company"],axis=1)
    
    target = 'top_rating'
    # creating classification column
    df[target] = np.where(df['rating'] > 4.5, 1, 0)
    
    # dropping score columns that are not the target    
    df = df.drop(["Downloads","numberreviews","five", "four", "three", "two", "one", "rating"],axis=1)
    
    result = naives_bayes_service.naives_bayes(df, 'top_rating')
    return result