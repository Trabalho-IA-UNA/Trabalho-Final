from application import decision_tree_appservice as tree_appservice
from application import k_algorithms_appservice as k_appservice
from application import regression_appservice as reg_appservice
from application import repository_appservice as repo_appservice

repo_appservice.clear_temp()

df = repo_appservice.get_treated_dataset()

kmeans = k_appservice.kmeans(df)