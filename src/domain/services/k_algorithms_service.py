import pandas as pd
import numpy as np
import matplotlib.lines
import matplotlib.markers
import matplotlib.colors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

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


def get_best_n_clusters(x, show_results = False) -> int:
    print('Calculando melhor n??mero de clusters...')
    # valores candidatos ao n?? de centroides
    parameters = [2, 3, 4, 5, 10, 15, 20]
    
    parameter_grid = ParameterGrid({'n_clusters': parameters})
    best_score = -1
    kmeans_model = KMeans()
    silhouette_scores = []
   
    # processa KMeans para cada op????o de n_clusters e atualiza o best_score
    for i, p in enumerate(parameter_grid, start=1):
        kmeans_model.set_params(**p)
        kmeans_model.fit(x)
        current_score = metrics.silhouette_score(x, kmeans_model.labels_)
        silhouette_scores += [current_score]
        # atualiza o maior valor
        if current_score > best_score:
            best_score = current_score
        # best_score = max([best_score, current_score])
            
        if show_results:
            print('Clusters:', p, 'Score', current_score)
        else:
            print(f'{i}/{len(parameters)}', end=', ')
    
    if show_results:
        # plotting silhouette score
        plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
        plt.xticks(range(len(silhouette_scores)), list(parameters))
        plt.title('Silhouette Score', fontweight='bold')
        plt.xlabel('Number of Clusters')
        plt.show()
    
    best_index = silhouette_scores.index(best_score)
    best_n_clusters = parameters[best_index]
    return best_n_clusters

def kmeans(dataset: pd.DataFrame, target: str, dimensions: tuple[str,str] | None, num_clusters: int | None):
    """Executa o algoritmo KMeans no dataframe especificado.

    Args:
        dataset (pd.DataFrame): O dataset contendo os dados.
        target (str): O nome da coluna da vari??vel target.
        dimensions (tuple[str,str] | None): O nome das colunas que representar??o as dimens??es X e Y. Caso seja none, ser?? aplicado o PCA para definir as dimens??es com maior peso.
        num_clusters (int | None): O n??mero de clusters (centr??ides) a serem criados. Caso seja none, ser??o feitos testes com os valores [2, 3, 4, 5, 10, 15, 20] e escolhido aquele que apresentar o maior score.
    """    
    
    # cria uma c??pia do dataset para transformar as escalas
    scaled_dataset = dataset[:].drop(['top_rating'], axis=1)
    
    # transforma as escalas para uma magnitude padr??o
    scaled_dataset[scaled_dataset.columns] = StandardScaler().fit_transform(scaled_dataset)
    
    _labels = ('','')
    _title = ''
    
    # se as dimens??es n??o forem especificadas, realizar PCA para obter as viari??veis de maior import??ncia (peso), unificando-as em 2 dimens??es
    if dimensions == None:
        pca_2 = PCA(n_components=2)
        pca_2_result = pca_2.fit_transform(scaled_dataset)
        # dataset_pca = pd.DataFrame(abs(pca_2.components_), columns=scaled_dataset.columns, index=['PC_1', 'PC_2'])
        # print(dataset_pca)
        ratios = pca_2.explained_variance_ratio_
        _labels = (f'PCA 1 ({ratios[0]})', f'PCA 2 ({ratios[1]})')
        _title = 'Agrupamento utilizando PCA'
        
        X = pca_2_result
    elif type(dimensions) == tuple:
        _labels = (dimensions[0], dimensions[1])
        _title = 'Agrupamento utilizando duas dimens??es'
        
        X = np.array(scaled_dataset[[dimensions[0], dimensions[1]]])
        
        
    # n?? de centr??ides especificado ou calculado
    n_centroides = num_clusters if num_clusters != None else get_best_n_clusters(X)
    # executa o algoritm KMeans
    kmeans = KMeans(n_clusters= n_centroides, random_state=0).fit(X)
    # obt??m os agrupamentos (grupos de centr??ides)
    agrupamento = kmeans.predict(X)
    
    # ----Exibindo Valores Agrupados----
    
    # exibe gr??fico em tela
    plot_agrupamento(X, agrupamento, kmeans.cluster_centers_, _labels, None, _title + f' ({n_centroides} centr??ides)')
    
    
    # ----Exibindo Valores Reais----
    
    _legends = ('Rating Abaixo 4.5', 'Rating Acima 4.5')
    _title = "Valores reais utilizando" + (" PCA" if 'PCA' in _labels[0] else " duas dimens??es")
    
    # substitui valores da coluna target por legenda amig??vel
    dataset[target].replace(0, _legends[0], inplace=True)
    dataset[target].replace(1, _legends[1], inplace=True)
    
    principal_df = pd.DataFrame(data = X, columns = _labels)
    
    # exibe gr??fico em tela
    plot_real(principal_df, dataset, target, _labels, _legends, _title)
    
    plt.show()
    

def plot_agrupamento(x: pd.DataFrame | np.ndarray, agrupamento: np.ndarray[int], centroides: list[tuple[int, int]], labels: tuple[str,str], legends: tuple[str,str] | None, title: str):
    """Exibe em tela um gr??fico de dispers??o, contentdo o n??mero de centroides (clusters) especificado.

    Args:
        x (pd.DataFrame | np.ndarray): Cole????o contendo os dados a serem plotados.
        agrupamento (np.ndarray[int]): Cole????o com o index do cluster que cada dado pertence. Resultado do KMeans.predict(x).
        centroides (list[tuple[int, int]]): Lista de coordenadas do centro de cada cluster.
        labels (tuple[str,str]): Legendas dos eixos X e Y do gr??fico.
        legends (tuple[str,str] | None): Legendas das cores dos elementos do gr??fico.
        title (str): T??tulo do gr??fico.
    """    
    plt.figure(figsize=(8,8))
    plt.axes()
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.title(title,fontsize=15)
    plt.xlabel(labels[0],fontsize=15)
    plt.ylabel(labels[1],fontsize=15)
    print(len(agrupamento))
    
    dots = plt.scatter(x[:, 0], x[:, 1], c=agrupamento, s=50, cmap='viridis')
    cluster_handle = None
    for x in centroides:
        cluster = plt.scatter(x[0], x[1], s=300, c=[0], cmap='gist_gray', edgecolors='white', alpha=0.3, marker='X')
        cluster_handle = cluster.legend_elements()[0]
    
    if legends != None:
        plt.legend(legends,prop={'size': 12})
    else:
        _legends = [f'Grupo {i+1}' for i in range(len(centroides))]
        _legends.append('Centr??ide')
        _handle = dots.legend_elements()[0]
        plt.legend(handles=[*_handle, *cluster_handle], labels=_legends, prop={'size': 12})
    

def plot_real(x: pd.DataFrame | np.ndarray, raw_df: pd.DataFrame , target: str, labels: tuple[str,str], legends: tuple[str,str] | None, title: str):
    """Exibe em tela um gr??fico de dispers??o, mapeando as classes que atendem ou n??o a condi????o da vari??vel target.

    Args:
        x (pd.DataFrame | np.ndarray): Cole????o contendo os dados a serem plotados.
        raw_df (pd.DataFrame): Dataframe cru, sem altera????es, para leitura dos registros que atendem ?? target.
        target (str): Nome da coluna da vari??vel target.
        labels (tuple[str,str]): Legendas dos eixos X e Y do gr??fico.
        legends (tuple[str,str] | None): Legendas das cores dos elementos do gr??fico.
        title (str): T??tulo do gr??fico.
    """    
    
    plt.figure(figsize=(8,8))
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.title(title,fontsize=15)
    plt.xlabel(labels[0],fontsize=15)
    plt.ylabel(labels[1],fontsize=15)
        
    colors = ['r', 'g']
    for lbl, color in zip(legends,colors):
        indices = raw_df[target] == lbl
        plt.scatter(x.loc[indices, labels[0]], x.loc[indices, labels[1]], c = color, s = 50)
        
    if legends != None:
        plt.legend(legends,prop={'size': 12})
    else:
        plt.legend(('Grupo 1','Grupo 2'),prop={'size': 12})

    
    
    
    