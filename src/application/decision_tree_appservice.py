import pandas as pd
import numpy as np
from sklearn import tree
import graphviz

from domain.utils import constants
from domain.models.results import DecisionTreeResult
from domain.services import decision_tree_service


def decision_tree(dataset: pd.DataFrame, tree_depth: int = 10, dataset_usage: float = 0.7,  open_file = False) -> DecisionTreeResult:
    """Executes Decision Tree on given DataFrame.

    Args:
        dataset (pd.DataFrame): The DataFrame containing the data to train on.
        tree_depth (int, optional): Number of tree nodes to generate from root. Defaults to 10.
        dataset_usage (float, optional): Percentage of the dataframe (0-1) to use for training. Defaults to 0.7.
        open_file (bool, optional): If the png exported tree file should be opened after creation. Defaults to False.

    Returns:
        DecisionTreeResult: _description_
    """
    
    # converts categories into columns (dummies)
    categories = pd.get_dummies(dataset["category"],drop_first=True)
    categories = pd.DataFrame.add_prefix(categories, 'category_')
    
    # drops text columns
    df = dataset.drop(["fancyname","category","company"],axis=1)
    
    # appending the encoded category column
    df = pd.concat([df, categories],axis=1)
    
    target = 'top_rating'
    # creating classification column
    df[target] = np.where(df['rating'] > 4.5, 1, 0)
    
    # dropping score columns that are not the target    
    df = df.drop(["Downloads","numberreviews","five", "four", "three", "two", "one", "rating"],axis=1)
    
    result = decision_tree_service.decision_tree(df,target, tree_depth, dataset_usage)
    
    # creates tree dot file
    dot_data = tree.export_graphviz(result.tree_model, out_file=None,
        feature_names=result.normalized_data.drop([target],axis=1).columns, 
        class_names= ['0','1'],
        filled=True, rounded=True,
        special_characters=True)
    
    # renders dot file as png image, saving in temp folder and opening, if open_file is True
    graph = graphviz.Source(dot_data, format='png')
    graph.render(filename="top_game_val", directory=constants.TEMP_PATH, view=open_file)
    
    return result
    
