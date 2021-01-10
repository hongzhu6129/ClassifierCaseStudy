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

#- dropping values equivalent to '?'
print("\n2. Information after Removed '?' data:")
data = data.dropna()
print("1.2. Data statistaics: ")
print(data.info())
print("1.3. Null data: ")
print(data.isnull().sum())

#Feature preprocessing
print("\n2. Feature preprocessing:")

#Use Label Encoder and Ordinal Encoder for Feature Processing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
ordinal_enc = OrdinalEncoder()
label_enc = LabelEncoder()
label_cat = data['class']
label_class = label_enc.fit_transform(label_cat).reshape(-1,1)
ord_cat = data.iloc[:,1:]
ord_cat.columns
ordinal_class = ordinal_enc.fit_transform(ord_cat)
print("label Cat:")
print(label_cat)
print("label Class:")
print(label_class)
print(ordinal_class[0:5])

#- Copy preprocessed data to a new data set
NewData1 = pd.DataFrame(ordinal_class,columns = ord_cat.columns)
NewData2 = pd.DataFrame(label_class,columns = ["class"])
NewData = pd.concat([NewData1,NewData2],axis=1)
print(NewData.columns)
NewData.info()
print(NewData.isnull().sum())

#Set lobal parameters
print("\n3. Set Global Parameters")

#- Split data set into train and test sets:
print("\n3.1. Split data set into train and test sets:")
from sklearn.model_selection import train_test_split
X = NewData.iloc[:,:-1]
y = NewData['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)

print("\n3.2. Set training and model parameters:")
#- Set training and model parameters:

output = 'class'
predict = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']

#Build models
print("\n4. Build Models")

#Build Random Forest models
#Random forest model (1)
print("\n--Random Forest (1)")
s=time.time()
rf = RandomForestClassifier(n_estimators=200)
rf = classification_model(rf, NewData, predict, output)
predictions = rf.predict(X_test)    
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(rf, open('RF_model.sav', 'wb'))

#Random forest model (2): Using Train Test Split
print("\n--Random Forest (2): Using Train Test Split")
s=time.time()
rf2 = RandomForestClassifier(n_estimators=200)
rf2 = classification_model2(rf2, X_train, predict, y_train)
predictions = rf2.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(rf2, open('RF2_model.sav', 'wb'))

