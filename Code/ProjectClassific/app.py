from ClassVar import AccesDb, OpsDb,Stemming, TfidfStopWord, CategoryID, ConfusionMatrix
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
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
import json

HOST = '0.0.0.0'
PORT = 8081
app  = Flask(__name__)
df = AccesDb()
df = CategoryID(df)
print ('Jumlah Dataset : ', len(df.index))
print('Jumlah Data Skripsi Title :', len(df['SkripsiTitle'].value_counts()))
    
CORS(app)
@app.route('/api/InsertData', methods=['POST'])
def InsertData():
    data = request.get_json()
    db = OpsDb()
    posts = db.Datasets
    post_id = posts.insert_one(data).inserted_id
    if post_id is None:
        result = [
            {'Result' : False}
        ]
        return jsonify(result)
    else:
        result = [
            {'Result' : True},
        ]
        return jsonify(result)


@app.route('/api/CheckData', methods=['GET'])
def CheckData():
    tempData = df['SkripsiCategory'].value_counts()
    Data = [{key:value} for key,value in tempData.items()]
    print(Data);
    return jsonify(Data)

@app.route('/api/train', methods=['POST'])
def TrainModel():
    parameters = request.get_json()
    
    features, labels, tfidf = SelectionFeatures(df)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, labels, test_size=0.2, random_state=0, shuffle=True)
    
    Model = LinearSVC(C=float(parameters['C']))
    ClfModel = Model.fit(Xtrain, Ytrain)
    y_true, y_pred = Ytest, Model.predict(Xtest)
    joblib.dump(ClfModel, 'Model.pkl')
    result = [  {'accuracy' : round(metrics.accuracy_score(y_true, y_pred) * 100, 2)}]
    return jsonify(result)

@app.route('/api/BuildClassification', methods=['POST'])
def BuildClassification():
    data = request.get_json()
    data = [(data['JudulSkripsi'])]
    print(data)
    category_id_df = df[['SkripsiCategory', 'category_id']].drop_duplicates().sort_values('category_id')
    id_to_category = dict(category_id_df[['category_id', 'SkripsiCategory']].values)
    ClfModel = joblib.load('Model.pkl')
    features, labels, tfidf = SelectionFeatures(df)
    TextFeature = tfidf.transform(data)
    predictions = ClfModel.predict(TextFeature)
    for text, predicted in zip(data, predictions):
        if id_to_category[predicted] == "Ai":
            str = "Artifical Inteligence"
            result = [
                {'CategorySkripsi' : "{}".format(str)}
            ]
        elif id_to_category[predicted] == "DS":
            str = "Data Science"
            result = [
                {'CategorySkripsi' : "{}".format(str)}
            ]
        elif id_to_category[predicted] == "IoT":
            str = "IoT"
            result = [
                {'CategorySkripsi' : "{}".format(str)}
            ]
        elif id_to_category[predicted] == "NS":
            str = "Network System"
            result = [
                {'CategorySkripsi' : "{}".format(str)}
            ]
        elif id_to_category[predicted] == "IS":
            str = "Information Security"
            result = [
                {'CategorySkripsi' : "{}".format(str)}
            ]
        elif id_to_category[predicted] == "SE":
            str = "Software Engineer"
            result = [
                {'CategorySkripsi' : "{}".format(str)}
            ]
        else:
            str = "Kategori Tidak Diketahui"
            result = [
                {'CategorySkripsi' : "{}".format(str)}
            ]

    return jsonify(result)


def SelectionFeatures(df):
    # ----- Stemming --------
    #df = Preprocessing(df)
    #------------------------

    #TFIDF
    tfidf = TfidfStopWord()
    start = timeit.default_timer()
    features = tfidf.fit_transform(df['SkripsiTitle']).toarray()
    labels = df.category_id

    stop = timeit.default_timer()
    runtime = stop - start
    print("Preprocessing and Selection Features  Process Executed in {0} Second !".format(runtime))
    return features, labels, tfidf

def Preprocessing(data):
    #Stemming Process
    start = timeit.default_timer()
    df = Stemming(data)
    stop = timeit.default_timer()
    runtime = stop - start
    print("Stemming Process Executed in {0} Second !".format(runtime))
    return df

if __name__ == '__main__':
    # run web server
    app.run(host=HOST,
            debug=True,  # automatic reloading enabled
            port=PORT)



