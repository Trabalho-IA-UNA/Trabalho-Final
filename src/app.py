from application import k_algorithms_appservice as k_appservice
from application import regression_appservice as reg_appservice
from application import repository_appservice as repo_appservice

df = repo_appservice.get_treated_dataset()

knn = k_appservice.knn(df)
knn.print_overview()