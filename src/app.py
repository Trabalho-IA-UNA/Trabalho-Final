from application import repository_appservice as repo_appservice

df = repo_appservice.get_treated_dataset(True)

print(df.info())

print(df.head())