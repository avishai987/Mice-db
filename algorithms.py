# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 15:51:24 2021

@author: אבישי
"""


#%% load data and make dataset
import pandas as pd
import numpy as np
from statistics import mean,variance
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

#cd "C:\\Users\\avish\\OneDrive\\מסמכים\\שנה ג' סמסטר ב\\פרויקט גמר\\פייתון\\Add_Hydrophobicity"
data_name = "trb_with_hyd"
path = "./pickles/" +data_name+".csv"
sample_clusters_hydro = pd.read_csv(path)
sample_mice = pd.read_csv('sample.csv')
mice_y = pd.read_csv('MICE.csv')

# make mice dic for making labels
mice_dic = {}
for index, row in mice_y.iterrows():
    mice_id = row['mice_id']
    survived = row['survived']
    if mice_id not in mice_dic:

        mice_dic[mice_id] = survived

#make labels
mice_y_list = []

for index, row in sample_mice.iterrows():  # iterate all rows
    mice_id = row['mice_id']
    mice_y_list.append(mice_dic[mice_id])
sample_mice['survived'] = mice_y_list

#drop control samples
for index, row in sample_mice.iterrows():
    survived = row['survived']
    mice_id = row['mice_id']
    if (survived == 'ctrl'):
        sample_mice = sample_mice.drop(index)
        sample_clusters_hydro = sample_clusters_hydro.drop(index)
mice_y_list = list(filter(('ctrl').__ne__, mice_y_list))


sample_clusters_hydro = sample_clusters_hydro.to_numpy()
labels = np.array(mice_y_list)
data = np.delete(sample_clusters_hydro, 0, 1) #delete sample id column
#%% dimensionality reduction
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron


X_embedded = TSNE(n_components=2).fit_transform(data)
x = X_embedded[:,0]
y = X_embedded[:,1]
perceptron_model = Perceptron(tol=1e-3)
perceptron_model.fit(X_embedded, labels)
arr = perceptron_model.coef_
a = arr[0,0]
b= arr[0,1]


cdict = {"non-responder": 'red', "responder": 'blue'}
fig, ax = plt.subplots()
plt.plot(x, a*x+b, '-g', label="y =" +str(float("{:.2f}".format(a)))+"x"+str(float("{:.2f}".format(b))))
for g in np.unique(labels):
    ix = np.where(labels == g)
    ax.scatter(x[ix], y[ix], c = cdict[g], label = g, s = 100)
ax.legend()
plt.show()

#%% Perceptron
from sklearn.linear_model import Perceptron

perceptron_model = Perceptron(tol=1e-3)
perceptron_model.fit(data, labels)

y_pred = perceptron_model.predict(data)
print( "perceptron f1 = " ,f1_score(labels,y_pred,average = 'weighted'))


    

#%% KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import utils
from sklearn.model_selection import StratifiedShuffleSplit    
  

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


(trainX, testX, trainY, testY) = train_test_split(data, labels,
 	test_size=0.25, stratify = labels)
parameters = {
       'n_neighbors':[1,3,5,7,9,11],
    }
gd = GridSearchCV(KNeighborsClassifier(), parameters)
gd.fit(trainX,trainY) 

k = gd.best_params_['n_neighbors']

f1 = 0
rounds = 100
for i in range (rounds):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
     	test_size=0.25, stratify = labels)
    knn_model.fit(trainX, trainY)
    y_pred = knn_model.predict(testX)
    f1 += f1_score(testY,y_pred,average = 'weighted')
print ("f1=", f1/rounds)



#%% Random forest
from sklearn.ensemble import RandomForestClassifier


f1 = 0
rounds = 50
for i in range (rounds):
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
 	test_size=0.25, stratify = labels)
    rf = RandomForestClassifier()
    rf.fit(trainX, trainY)
    y_pred = rf.predict(testX)
    f1 += f1_score(testY,y_pred,average = 'weighted')
print ("f1=", f1/rounds)


#%%  xgboost
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(labels)

f1 = 0
rounds = 50
for i in range (rounds):
    (trainX, testX, trainY, testY) = train_test_split(data, encoded_labels,
     	test_size=0.25, stratify = encoded_labels)
        
    train = xgb.DMatrix(trainX, label= trainY )
    test =  xgb.DMatrix(testX, label= testY )
    params = {
        'max_depth':5,
        'eta' : 0.3,
        'objective': 'multi:softmax',
        'num_class': 2
        }
    epochs = 50
    model = xgb.train(params,train,epochs)
    y_pred = model.predict(test)
    f1+= f1_score(testY,y_pred,average = 'weighted')
print ("f1= ",f1/rounds)

#%% Perceptron with feature selection
from sklearn.linear_model import Perceptron
from sklearn.feature_selection import SequentialFeatureSelector
all_f1 = []
for i in range (1,10):
    perceptron_model = Perceptron(tol=1e-3)
    sfs = SequentialFeatureSelector(perceptron_model, n_features_to_select=i,scoring='f1_weighted')
    a = sfs.fit(data, labels)
    new_data = sfs.transform(data) 
    perceptron_model.fit(new_data, labels)
    
    y_pred = perceptron_model.predict(new_data)
    all_f1.append(f1_score(labels,y_pred,average = 'weighted'))
print(max(all_f1))
#%% KNN feature selection 
import sklearn.feature_selection 
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
for i in range (1,10):
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
 	test_size=0.25, stratify = labels)
    parameters = {
       'n_neighbors':[1,3,5,7,9,11],
    }
    gd = GridSearchCV(KNeighborsClassifier(), parameters)
    gd.fit(trainX,trainY) 

    k = gd.best_params_['n_neighbors']
    
    knn = KNeighborsClassifier(n_neighbors=k)

    sfs = SequentialFeatureSelector(knn, n_features_to_select=i,scoring='f1_weighted')
    a = sfs.fit(data, labels)
    new_data = sfs.transform(data)  
    
    f1 = 0
    rounds = 100
    for j in range (rounds):
        
        (trainX, testX, trainY, testY) = train_test_split(new_data, labels,
         	test_size=0.25, stratify = labels)
        knn.fit(trainX, trainY)
    
        y_pred = knn.predict(testX)
        f1 += f1_score(testY,y_pred,average = 'weighted')
    print ("i =", i, "f1=", f1/rounds)
#%% dimensionality reduction with feature selection 
X_embedded = TSNE(n_components=2).fit_transform(new_data)
x = X_embedded[:,0]
y = X_embedded[:,1]
cdict = {"non-responder": 'red', "responder": 'blue'}
fig, ax = plt.subplots()
for g in np.unique(labels):
    ix = np.where(labels == g)
    ax.scatter(x[ix], y[ix], c = cdict[g], label = g, s = 100)
ax.legend()
plt.show()



#%% Perceptron with feature selection
from sklearn.linear_model import Perceptron
from sklearn.feature_selection import SequentialFeatureSelector
for i in range (2,10):
    perceptron_model = Perceptron(tol=1e-3)
    sfs = SequentialFeatureSelector(perceptron_model, n_features_to_select=i,scoring='f1_weighted')
    a = sfs.fit(data, labels)
    new_data = sfs.transform(data) 

    
    X_embedded = TSNE(n_components=2).fit_transform(new_data)
    x = X_embedded[:,0]
    y = X_embedded[:,1]
    perceptron_model.fit(X_embedded, labels)
    arr = perceptron_model.coef_
    a = arr[0,0]
    b= arr[0,1]
    cdict = {"non-responder": 'red', "responder": 'blue'}
    fig, ax = plt.subplots()
    plt.plot(x, a*x+b, '-g', label="y =" +str(float("{:.2f}".format(a)))+"x"+str(float("{:.2f}".format(b))))

    for g in np.unique(labels):
        ix = np.where(labels == g)
        ax.scatter(x[ix], y[ix], c = cdict[g], label = g, s = 100)
    ax.legend()
    plt.show()


    

