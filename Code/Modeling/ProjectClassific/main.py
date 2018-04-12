from io import StringIO
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import numpy as np
from StopWordID import get_stop
import matplotlib.pyplot as plt
import pprint
import pandas as pd
import seaborn as sns
import timeit
from ClassVar import AccesDb

#fig = plt.figure(figsize=(8, 6))
# IDStopWords = set(get_stop())
# tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=IDStopWords)
# features = tfidf.fit_transform(df.SkripsiTitle).toarray()
# labels = df.category_id

def main():
    df = AccesDb()
    Preprocessing(df)
    #CheckData(df)
    

def CheckData(data):
    #print(data.head())
    print ('Jumlah Dataset : ', len(data.index))
    print('Jumlah Data Skripsi Title', len(data['SkripsiTitle'].value_counts()))
    print('Jumlah Data Skripsi Abstrak', len(data['SkripsiAbstrak'].value_counts()))
    data.groupby('SkripsiCategory').SkripsiTitle.count().plot.bar(ylim = 0)
    plt.show()

def Preprocessing(data):
    #Create category ID
    df = CategoryID(data)
    category_id_df = df[['SkripsiCategory', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'SkripsiCategory']].values)
    print(df.head())

def CategoryID(data):
    col = ['SkripsiCategory', 'SkripsiTitle']
    df = data[col]
    print(df.columns)
    df.columns = ['SkripsiCategory', 'SkripsiTitle']
    df['category_id'] = df['SkripsiCategory'].factorize()[0]
    return df

if __name__ == '__main__':
    main()