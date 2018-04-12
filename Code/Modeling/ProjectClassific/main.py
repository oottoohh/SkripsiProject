from io import StringIO
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
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
import matplotlib.pyplot as plt
import pprint
import pandas as pd
import seaborn as sns
import timeit
from ClassVar import AccesDb, Stemming, StopWord

#fig = plt.figure(figsize=(8, 6))

def main():
    df = AccesDb()
    #CheckData(df)
    Preprocessing(df)
    
    

def CheckData(data):
    #print(data.head())
    print ('Jumlah Dataset : ', len(data.index))
    print('Jumlah Data Skripsi Title :', len(data['SkripsiTitle'].value_counts()))
    print('Jumlah Data Skripsi Abstrak :', len(data['SkripsiAbstrak'].value_counts()))
    print('Jumlah Data Category :\n', data.SkripsiCategory.value_counts())
    data.groupby('SkripsiCategory').SkripsiTitle.count().plot.bar(ylim = 0)
    plt.show()

def Preprocessing(data):
    df = Stemming(data)
    print(' ----- Hasil Stemming ----- \n ', df['SkripsiTitle'].head())
    df = CategoryID(data)
    category_id_df = df[['SkripsiCategory', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'SkripsiCategory']].values)
    print(df.head())
    tfidf = StopWord()
    features = tfidf.fit_transform(df['SkripsiTitle']).toarray()
    labels = df.category_id
    print(features.shape)

def CategoryID(data):
    col = ['SkripsiCategory', 'SkripsiTitle']
    df = data[col]
    print(df.columns)
    df.columns = ['SkripsiCategory', 'SkripsiTitle']
    df['category_id'] = df['SkripsiCategory'].factorize()[0]
    return df

if __name__ == '__main__':
    main()