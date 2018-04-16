from pymongo import MongoClient
from pandas.io.json import json_normalize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from StopWordID import get_stop
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

def AccesDb():
    client = MongoClient()
    client = MongoClient('localhost', 27017)
    db = client['RepositoryUIN']
    dt= db.Datasets
    ListData = (list(dt.find()))
    df = json_normalize(ListData)
    return df

def Stemming(df):
    df = CategoryID(df)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    df['Tokenized'] = df['SkripsiTitle'].apply(lambda x : filter(None,x.split(" ")))
    df['TokenizeData'] = df['Tokenized'].apply(lambda x: [stemmer.stem(y) for y in x])
    df['SkripsiTitle'] = df['TokenizeData'].apply(lambda x : " ".join(x))
    return df

def TfidfStopWord():
    IDStopWords = set(get_stop())
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=IDStopWords) 
    return tfidf

def CategoryID(data):
    col = ['SkripsiCategory', 'SkripsiTitle']
    df = data[col]
    df.columns = ['SkripsiCategory', 'SkripsiTitle']
    df['category_id'] = df['SkripsiCategory'].factorize()[0]
    return df

def ConfusionMatrix(Ytest, y_pred, df):
    df = CategoryID(df)
    category_id_df = df[['SkripsiCategory', 'category_id']].drop_duplicates().sort_values('category_id')
    conf_mat = confusion_matrix(Ytest, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=category_id_df.SkripsiCategory.values, yticklabels=category_id_df.SkripsiCategory.values )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()