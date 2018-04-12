from pymongo import MongoClient
from pandas.io.json import json_normalize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from StopWordID import get_stop
from sklearn.feature_extraction.text import TfidfVectorizer
def AccesDb():
    client = MongoClient()
    client = MongoClient('localhost', 27017)
    db = client['RepositoryUIN']
    dt= db.Datasets
    ListData = (list(dt.find()))
    df = json_normalize(ListData)
    return df

def Stemming(df):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    df = df.SkripsiTitle.astype(str).str.split(',')
    df["SkripsiTitle"] = df.apply(lambda x: [stemmer.stem(y) for y in x])
    df["SkripsiTitle"] = df['SkripsiTitle'].apply(lambda x : " ".join(x))
    return df

def StopWord():
    IDStopWords = set(get_stop())
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=IDStopWords)
    return tfidf
