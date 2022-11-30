import pandas as pd
import numpy as np
from sklearn import preprocessing


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
    
    
def kmeans(dataset: pd.DataFrame, num_clusters:int | None = None):
    """Executa o algoritmo KMeans no dataframe especificado.

    Args:
        dataset (pd.DataFrame): O dataset contendo os dados.
        num_clusters (int | None, optional): O número de clusters (centróides) a serem criados. Caso seja none, serão feitos testes com os valores [2, 3, 4, 5, 10, 15, 20] e escolhido aquele que apresentar o maior score. Defaults to None.
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
    
    # if score columns (that are not game features) should be included
    include_scores = False
    
    # dropping score columns that are not the target 
    if not include_scores:
        df = df.drop(["Downloads","numberreviews","five", "four", "three", "two", "one"],axis=1)
    
    # exemplos para testar
    dimension_examples = [
        None, # Será calculado PCA. Testar com 'include_scores' True e False.
        ('price', 'rating'), # Relação de avaliação por preço
        ('usersinteract','category'), # Quais categorias têm uma boa aceitação nos jogos multiplayer
        ('numberreviews','price'), # Quantidade de avaliações por preço. 'include_scores' deve ser True
        ('numberreviews','five'), # Relação de 5 estrelas para quantidade de avaliações. 'include_scores' deve ser True
    ]
    
    k_algorithms_service.kmeans(df, 'top_rating', dimensions= dimension_examples[0], num_clusters= num_clusters)