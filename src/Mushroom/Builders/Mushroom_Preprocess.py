#Import packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import pickle

#Definition model builder functions

from sklearn.linear_model import LogisticRegression   
from sklearn.model_selection import KFold 
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

def classification_model(model, data, predictors, outcome):  
    #Fit the model:  
    model.fit(data[predictors],data[outcome])    
    #Make predictions on training set:  
    predictions = model.predict(data[predictors])    
    #Print accuracy  
    accuracy = metrics.accuracy_score(predictions,data[outcome])  
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
    #Perform k-fold cross-validation with 10 folds  
    kf = KFold(10,shuffle=True)  
    error = []  
    for train, test in kf.split(data):
        # Filter training data    
        train_predictors = (data[predictors].iloc[train,:])        
        # The target we're using to train the algorithm.    
        train_target = data[outcome].iloc[train]        
        # Training the algorithm using the predictors and target.    
        model.fit(train_predictors, train_target)
        #Record error from each cross-validation run    
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
     
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))) 
    # %s is placeholder for data from format, next % is used to conert it into percentage
    #.3% is no. of decimals
    return model

def classification_model2(model, x_train, predictors, y_train):
    #Fit the model:  
    model.fit(x_train[predictors], y_train)    
    #Make predictions on training set:  
    predictions = model.predict(x_train[predictors])    
    #Print accuracy  
    accuracy = metrics.accuracy_score(predictions, y_train)  
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
    #Perform k-fold cross-validation with 10 folds  
    kf = KFold(10,shuffle=True)  
    error = []  
    for train, test in kf.split(x_train):
        # Filter training data    
        train_predictors = (x_train[predictors].iloc[train,:])        
        # The target we're using to train the algorithm.    
        train_target = y_train.iloc[train]      
        # Training the algorithm using the predictors and target.    
        model.fit(train_predictors, train_target)
        #Record error from each cross-validation run    
        error.append(model.score(x_train[predictors].iloc[test,:], y_train.iloc[test]))
     
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))) 
    # %s is placeholder for data from format, next % is used to conert it into percentage
    #.3% is no. of decimals
    return model

#Load data from .csv file

data = pd.read_csv("C:\Morphy\Mushroom\mushrooms.csv")

print("1. Information of Raw Data:")
print("1.1. Data Head: ")
print(data.head())
print("1.2. Data statistaics: ")
print(data.info())
print("1.3. Null data: ")
print(data.isnull().sum())

# -- Note: missing data in stalk-root feature is represented in the form of '?'.
#They are replaced by the standard representation of np.NaN

data['stalk-root'].replace({'?':np.NaN},inplace=True)
#To see how many missing data in the dataset:
print("1.4. Null data after repplacing '?' with np.NaN: ")
print(data.isnull().sum())

#The rows with missing data are removed. 
data = data.dropna()

#New statistics
print("1.5. Data statistaics: ")
print(data.info())

#Feature preprocessing
print("\n2. Feature preprocessing:")

#Encoding label and feature categories 
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OrdinalEncoder
#label_enc = LabelEncoder()
#label_cat = data['class']
#label_class = label_enc.fit_transform(label_cat).reshape(-1,1)

data['class'].replace({"e":0,"p":1},inplace=True)
data['cap-shape'].replace({'b':0, 'c':1, 'f':2, 'k':3, 's':4, 'x':5}, inplace=True)
data['cap-surface'].replace({'f':0, 'g':1, 's':2, 'y':3}, inplace=True)
data['cap-color'].replace({'b':0, 'c':1, 'e':2, 'g':3, 'n':4, 'p':5, 'r':6, 'u':7, 'w':8, 'y':9}, inplace=True)
data['bruises'].replace({'f':0, 't':1}, inplace=True)
data['odor'].replace({'a':0, 'c':1, 'f':2, 'l':3, 'm':4, 'n':5, 'p':6, 's':7, 'y':8}, inplace=True)
data['gill-attachment'].replace({'a':0, 'd':1, 'f':2, 'n':3},inplace=True)
data['gill-spacing'].replace({'c':0, 'd':1, 'w':2},inplace=True)
data['gill-size'].replace({'b':0, 'n':1},inplace=True)
data['gill-color'].replace({'b':0, 'e':1, 'g':2, 'h':3, 'k':4, 'n':5, 'o':6, 'p':7, 'r':8, 'u':9, 'w':10, 'y':11},inplace=True)
data['stalk-shape'].replace({'e':0, 't':1},inplace=True)
data['stalk-root'].replace({'b':0, 'c':1, 'e':2, 'r':3, 'u':4, 'z':5},inplace=True)
data['stalk-surface-above-ring'].replace({'f':0, 'k':1, 's':2, 'y':3},inplace=True)
data['stalk-surface-below-ring'].replace({'f':0, 'k':1, 's':2, 'y':3},inplace=True)
data['stalk-color-above-ring'].replace({'b':0, 'c':1, 'e':2, 'g':3, 'n':4, 'o':5, 'p':6, 'w':7, 'y':8},inplace=True)
data['stalk-color-below-ring'].replace({'b':0, 'c':1, 'e':2, 'g':3, 'n':4, 'o':5, 'p':6, 'w':7, 'y':8},inplace=True)
data['veil-type'].replace({'p':0, 'u':1}, inplace=True)
data['veil-color'].replace({'n':0, 'o':1, 'w':2, 'y':3},inplace=True)
data['ring-number'].replace({'n':0, 'o':1, 't':3},inplace=True)
data['ring-type'].replace({'c':0, 'e':1, 'f':2, 'l':2, 'n':3, 'p':4, 's':5, 'z':6},inplace=True)
data['spore-print-color'].replace({'b':0, 'h':1, 'k':2, 'n':3, 'o':4, 'r':5, 'u':6, 'w':7, 'y':8},inplace=True)
data['population'].replace({'a':0, 'c':1, 'n':2, 's':3, 'v':4, 'y':5},inplace=True)
data['habitat'].replace({'d':0, 'g':1, 'l':2, 'm':3, 'p':4, 'u':5, 'w':6},inplace=True)


print("2.1. Data statistaics after encoding: ")
print(data.info())


#ordinal_enc = OrdinalEncoder()
#ord_cat = data.iloc[:,1:]
#print("ord_cat:")
#print(ord_cat)
#ord_cat.columns
#print("ord_cat columns:")
#print(ord_cat.columns)
#ordinal_class = ordinal_enc.fit_transform(ord_cat)
#print("ordinal_class:")
#print(ordinal_class)
#print("cat inverse:")

#print(ordinal_class[0:5])

#- Copy preprocessed data to a new data set
#NewData1 = pd.DataFrame(ordinal_class,columns = ord_cat.columns)
#NewData2 = pd.DataFrame(label_class,columns = ["class"])
#NewData = pd.concat([NewData1,NewData2],axis=1)
#print(NewData.columns)
#NewData.info()
#print(NewData.isnull().sum())




