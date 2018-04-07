from pymongo import MongoClient
from pandas.io.json import json_normalize
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
import pymongo
import pprint
import pandas as pd
import seaborn as sns
import timeit

#check data
client = MongoClient()
client = MongoClient('localhost', 27017)
db = client['RepositoryUIN']
dt= db.Datasets
ListData = (list(dt.find()))
df = json_normalize(ListData)
#check all data
#print(df.head())

#check null data
df = df[pd.notnull(df['SkripsiTitle'])]
# print(df.info())

#cleansing column 
col = ['SkripsiCategory', 'SkripsiTitle']
df = df[col]
# print(df.columns)

#Create ID Category
df.columns = ['SkripsiCategory', 'SkripsiTitle']
df['category_id'] = df['SkripsiCategory'].factorize()[0]
category_id_df = df[['SkripsiCategory', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'SkripsiCategory']].values)
# print(df.head())

# Show Distribution Data
# fig = plt.figure(figsize=(8,6))
# df.groupby('SkripsiCategory').SkripsiTitle.count().plot.bar(ylim=0)
# plt.show()

# Preprocessing & Feature Extraction
IDStopWords = set(get_stop())
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=IDStopWords)
features = tfidf.fit_transform(df.SkripsiTitle).toarray()
labels = df.category_id
#print(features.shape)

#terms that are the most correlated with each of the SkripsiCategory

# N = 2
# for SkripsiCategory, category_id in sorted(category_to_id.items()):
#     features_chi2 = chi2(features, labels == category_id)
#     indices = np.argsort(features_chi2[0])
#     features_names = np.array(tfidf.get_feature_names())[indices]
#     unigrams = [v for v in features_names if len(v.split(' ')) == 1]
#     bigrams = [v for v in features_names if len(v.split(' ')) == 2]
#     print("# '{}' :".format(SkripsiCategory))
#     print("     . Most Correlated unigrams:\n   . {}".format('\n    .'.join(unigrams[-N:])))
#     print("     . Most Correlated bigrams:\n   . {}".format('\n    .'.join(bigrams[-N:])))    


#Selection Model
# models = [
#     RandomForestClassifier(n_estimators=50, max_depth=3, random_state=0),
    
#     LinearSVC(),
#     MultinomialNB(),
#     LogisticRegression(random_state=0),
# ]
# CV = 3
# cv_df = pd.DataFrame(index=range(CV * len(models)))
# entries = []
# for model in models:
#     model_name = model.__class__.__name__
#     accuracies = cross_val_score(model, features, labels, scoring='accuracy',cv=CV)
#     for fold_idx, accuracy in enumerate(accuracies):
#         entries.append((model_name, fold_idx, accuracy))
# cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

# sns.boxplot(x='model_name', y='accuracy', data=cv_df)
# sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor="gray", linewidth=2)
# print(cv_df.groupby('model_name').accuracy.mean())



# Split the dataset in two equal parts
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, shuffle=True, random_state=0)
# #Set the parameters by cross-validation
# parameters = [{'kernel': ['rbf'],
#                'gamma': [100000, 10000, 1000, 100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
#                 'C': [100000, 10000, 1000, 100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]},
#                {'kernel': ['linear'], 'C': [1000, 100, 10, 1, 0.9, 0.7, 0.4, 0.3, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
#             ]
parameters = [
                {'kernel': ['linear'], 'C': [0.7]}
             ]
# print("# Tuning hyper-parameters")
# print()

# clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
# clf.fit(X_train, y_train)

# print("Best parameters set found on development set:")
# print()
# print(clf.best_params_)
# print()
# print("Grid scores on training set:")
# print()
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))
# print()
# print(timeit.timeit())
# #-----------------#


#modeling
model = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
print(model.fit(features, labels))
y_pred = model.predict(X_test)
# conf_mat = confusion_matrix(y_test, y_pred)
# fig, ax = plt.subplots(figsize=(8,6))
# sns.heatmap(conf_mat, annot=True, fmt='d',
#             xticklabels=category_id_df.SkripsiCategory.values, yticklabels=category_id_df.SkripsiCategory.values)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()

#result Accuracy Score
print(metrics.classification_report(y_test, y_pred, 
                                    target_names=df['SkripsiCategory'].unique()))