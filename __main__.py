from src.application import decision_tree_appservice as tree_app_service
from src.application import k_algorithms_appservice as k_app_service
from src.application import regression_appservice as reg_app_service
from src.application import repository_appservice as repo_app_service

repo_app_service.clear_temp()

df = repo_app_service.get_treated_dataset()

tree = tree_app_service.decision_tree(df, open_file=True)
tree.print_overview()
