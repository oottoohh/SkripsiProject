from pymongo import MongoClient
from pandas.io.json import json_normalize
def AccesDb():
    client = MongoClient()
    client = MongoClient('localhost', 27017)
    db = client['RepositoryUIN']
    dt= db.Datasets
    ListData = (list(dt.find()))
    df = json_normalize(ListData)
    return df