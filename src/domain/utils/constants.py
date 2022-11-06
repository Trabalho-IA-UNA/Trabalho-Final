DATA_PATH = 'src\\infrastructure\\data'
"""The project relative path to the folder 'data' in infrastructure."""

DATASET_PATH = f'{DATA_PATH}\\android_apps_metadata.csv'
"""The project relative path to the original dataset CSV file."""

DATASET_SEPARATOR = ';'
"""The separator used in the original dataset CSV file."""

"""The 16 dataset columns that were chosen for the project scope."""
SCOPE_COLUMNS = [
    'fancyname',
    'category', 
    'company', 
    'purchases', 
    'ads', 
    'usersinteract', 
    'age_rating',
    'Downloads',
    'price',
    'rating',
    'numberreviews',
    'five',
    'four',
    'three',
    'two',
    'one'
    ]
