from ClassVar import AccesDb, Stemming, TfidfStopWord, CategoryID, ConfusionMatrix
from flask import Flask, request, jsonify
from io import StringIO
from sklearn import svm
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import pprint
import pandas as pd
import seaborn as sns
import timeit
import time


def main():
    df = AccesDb()
    # texts = ["Aplikasi Keamanan Data Multimedia Message Service (MMS) pada Microsoft Office File Memanfaatkan Algoritma Rivest-Shamir Adleman (RSA) dan Blowfish Berbasis Android",
    #         "Rancang Bangun Penerima Pemantau Cairan Infus Berbasis Mikrokontroler Menggunakan Media Komputer"]
    #CheckData(df)
    StartTime = time.time()
    #SelectionModel(df)
    
    start = timeit.default_timer()
    # print ('-------- The SVM model will begin the training process--------')    
    # ClfModel, tfidf, id = TrainModel(df)
    TrainModel(df)
    stop = timeit.default_timer()
    runtime = stop - start
    print (' ----- The SVM model has finished training in {0} second ----- !'.format(runtime))
    
    # start = timeit.default_timer()
    # BuildClassification(ClfModel, texts, tfidf, id)
    # stop = timeit.default_timer()
    # runtime = stop - start
    # print (' ----- Machine Learning has been Finishied learn Data in {0} second ----- !'.format(runtime))
    
    print ('The Script Executed {0} second !'.format(time.time() - StartTime))
    

def BuildClassification(ClfModel, data, tfidf, id):
    TextFeature = tfidf.transform(data)
    predictions = ClfModel.predict(TextFeature)
    for text, predicted in zip(data, predictions):
        print('"{}"'.format(text))
        print(" - DiPrediksi Kategori Skripsi Diatas adalah : '{}'".format(id[predicted]))
        print("")

def CheckData(data):
    #print(data.head())
    print ('Jumlah Dataset : ', len(data.index))
    print('Jumlah Data Skripsi Title :', len(data['SkripsiTitle'].value_counts()))
    print('Jumlah Data Skripsi Abstrak :', len(data['SkripsiAbstrak'].value_counts()))
    print('Jumlah Data Category :\n', data.SkripsiCategory.value_counts())
    data.groupby('SkripsiCategory').SkripsiTitle.count().plot.bar(ylim = 0)
    counts = data['SkripsiCategory'].value_counts()
    result = [{key:value} for key,value in counts.items()]
    print(result)
    #plt.show()



def SelectionFeatures(data):
    # ----- Stemming --------
    df = Preprocessing(data)
    #------------------------

    #Without Stemming
    #df = CategoryID(data)
    #TFIDF
    tfidf = TfidfStopWord()
    start = timeit.default_timer()
    features = tfidf.fit_transform(df['SkripsiTitle']).toarray()
    labels = df.category_id

    # Show Tfidf Scores Matrix
    # featuresName = tfidf.get_feature_names()
    # indexTitle = [n for n in df['SkripsiTitle']]
    # rows, cols = features.nonzero()
    # for row, col in zip(rows, cols):
    #     print(row, ".--- Hasil TFIDF ---", (featuresName[col], indexTitle[row]), features[row, col])
    # print(features.shape)

    #Menemukan Korelasi yang berkaitan dengan SkripsiCategory menggunakan Chi Square Unigram Bigram
    #category_id_df = df[['SkripsiCategory', 'category_id']].drop_duplicates().sort_values('category_id')
    # category_to_id = dict(category_id_df.values)    
    # N = 2
    # for SkripsiCategory, category_id in sorted(category_to_id.items()):
    #     features_chi2 = chi2(features, labels == category_id)
    #     indices = np.argsort(features_chi2[0])
    #     features_names = np.array(tfidf.get_feature_names())[indices]
    #     unigrams = [v for v in features_names if len(v.split(' ')) == 1]
    #     bigrams = [v for v in features_names if len(v.split(' ')) == 2]
    #     print("*'{}'* : ".format(SkripsiCategory))
    #     print(" . Kata yang Berelasi (unigrams) : \n    . {}".format('\n    .'.join(unigrams[-N:])))
    #     print(" . Kata yang Berelasi (Bigrams) : \n     . {}".format('\n    .'.join(bigrams[-N:])))

    stop = timeit.default_timer()
    runtime = stop - start
    print("Preprocessing and Selection Features  Process Executed in {0} Second !".format(runtime))
    return features, labels, tfidf

def Preprocessing(data):
    #Stemming Process
    df = Stemming(data)
    start = timeit.default_timer()
    print(' ----- Hasil Stemming ----- \n ', df.head())
    stop = timeit.default_timer()
    runtime = stop - start
    print("Stemming Process Executed in {0} Second !".format(runtime))
    return df

def SelectionModel(data):
    features, labels, tfidf = SelectionFeatures(data)
    Models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(Models)))
    Entries = []
    for model in Models:
        ModelName = model.__class__.__name__
        AccuracyData = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
        for foldIDX, accuracy in enumerate(AccuracyData):
            Entries.append((ModelName, foldIDX, accuracy))
    cv_df = pd.DataFrame(Entries, columns=['ModelName', 'foldIDX', 'accuracy'])
    sns.boxplot(x='ModelName', y='accuracy', data=cv_df)
    sns.stripplot(x='ModelName', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor='gray', linewidth=2)
    print(cv_df.groupby('ModelName').accuracy.mean()) 
    plt.show()

def TrainModel(data):
    df = CategoryID(data)
    category_id_df = df[['SkripsiCategory', 'category_id']].drop_duplicates().sort_values('category_id')
    id_to_category = dict(category_id_df[['category_id', 'SkripsiCategory']].values)
    
    features, labels, tfidf = SelectionFeatures(data)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, labels, test_size=0.2, random_state=0, shuffle=True)

    #TuningModel(Xtrain, Ytrain)

    Model = LinearSVC(C=0.9)
    Model.fit(Xtrain, Ytrain)
    y_true, y_pred = Ytest, Model.predict(Xtest)
    # Ypred = Model.predict(Xtest)
    # ConfusionMatrix(Ytest, Ypred, data)
    ClfModel = Model.fit(features, labels)
    
    #Show Top Gram with Model
    category_to_id = dict(category_id_df.values)
    TopGram(ClfModel, category_to_id, tfidf)
    
    print('--------------Specification Model ! --------------\n', ClfModel)
    print(metrics.classification_report(y_true, y_pred, target_names=df['SkripsiCategory'].unique()))
    print('Akurasi Model', round(metrics.accuracy_score(y_true, y_pred) * 100, 2), '%')
    return ClfModel, tfidf, id_to_category

def TopGram(Model, id, tfidf):
    N = 2
    for SkripsiCategory, category_id in sorted(id.items()):
        indices = np.argsort(Model.coef_[category_id])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
        bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
        print("# '{}':".format(SkripsiCategory))
        print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
        print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))

def TuningModel(X, y):
    parameters = [{'kernel': ['rbf'],
               'gamma': [100000, 10000, 1000, 100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                'C': [100000, 10000, 1000, 100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]},
               {'kernel': ['linear'], 'C': [1000, 100, 10, 1, 0.9, 0.7, 0.4, 0.3, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
            ]


    print("# Tuning hyper-parameters")
    print()

    clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on training set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
    print()

if __name__ == '__main__':
    main()