from application import repository_appservice as repo_appservice, regression_appservice

df = repo_appservice.get_treated_dataset()

logistic = regression_appservice.logistic(df)
logistic.print_overview()