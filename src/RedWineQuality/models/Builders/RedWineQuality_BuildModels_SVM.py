import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import os
import pickle

#Load data from .csv file
data = pd.read_csv("C:/Users/Hong/Downloads/winequality-red.csv")
data.info()

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

def classification_model2(model, x_train, p, y_train):#, outcome):  
    #Fit the model:  
    model.fit(x_train[p], y_train)    
    #Make predictions on training set:  
    predictions = model.predict(x_train[p])    
    #Print accuracy  
    accuracy = metrics.accuracy_score(predictions, y_train)  
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))
    #Perform k-fold cross-validation with 10 folds  
    kf = KFold(10,shuffle=True)  
    error = []  
    for train, test in kf.split(x_train):
        # Filter training data    
        train_predictors = (x_train[p].iloc[train,:])        
        # The target we're using to train the algorithm.    
        train_target = y_train.iloc[train]        
        # Training the algorithm using the predictors and target.    
        model.fit(train_predictors, train_target)
        #Record error from each cross-validation run    
        error.append(model.score(x_train[p].iloc[test,:], y_train.iloc[test]))
     
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))) 
    # %s is placeholder for data from format, next % is used to conert it into percentage
    #.3% is no. of decimals
    return model

#Split data set into train and test sets:

#Set training and model parameters:

from sklearn.model_selection import train_test_split
X = (data.iloc[:,0:11])
y = (data['quality'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)
output = 'quality'
#predict = ['alcohol','volatile acidity','sulphates','citric acid','residual sugar','pH']
predict = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

#Build Surport Vector Machine models

#Surport Vector Model (1)
print("\n--Surportting vector machine (1)")
s=time.time()
svc = svm.SVC()
svc = classification_model(svc,data,predict,output)
predictions = svc.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions,y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(svc, open('SVM_model.sav', 'wb'))

#Support vector machine (2): using Train Test split
print("\n--Surportting vector machine (2): Using Train Test split")
s=time.time()
svc2 = svm.SVC()
svc2 = classification_model2(svc2, X_train, predict, y_train)
predictions = svc2.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(svc2, open('SVM2_model.sav', 'wb'))

