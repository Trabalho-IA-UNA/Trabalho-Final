from application import repository_appservice as repo_appservice, regression_appservice

df = repo_appservice.get_treated_dataset()

linear = regression_appservice.linear(df)
linear.print_overview()