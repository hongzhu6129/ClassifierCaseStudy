#This Python program builds the following machine learning models:
#1: Logistic Regression models: LR_model.sav, LR2_model.sav
#2: K-Nearest Neighbors Models: KNN_model.sav, KNN2_model.sav
#3: Decision Tree Classifier models: DT_model.sav, DT2_model.sav
#4: Random forest model: RF_model.sav, RF2_model.sav
#5: Naive Bayes model: NB_model.sav, NB2_model.sav
#6: Ensemble learning models:
#     Soft Voting: SV_model.sav, SV2_model,sav
#     Hard Voting: HV_model.sav, HV2_model.sav
#     Stacking: Stack1_model.sav, Stack2_model.sav, Stack3_model.sav

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
print("1.5. New data statistaics: ")
print(data.info())

#Feature preprocessing
print("\n2. Feature preprocessing:")

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

#- Copy preprocessed data to a new data set
NewData1 = pd.DataFrame(data,columns = data.iloc[:,1:].columns)
NewData2 = pd.DataFrame(data,columns = ["class"])
NewData = pd.concat([NewData1,NewData2],axis=1)

print('2.2. Data columes:')
print(NewData.columns)
print('2.3. Data info:')
NewData.info()
print('2.3. Null data:')
print(NewData.isnull().sum())

#Set global parameters
print("\n3. Set Global Parameters")

#- Split data set into train and test sets:
print("\n3.1. Split data set into train and test sets:")
from sklearn.model_selection import train_test_split
X = NewData.iloc[:,:-1]
y = NewData['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)

#- Set training and model parameters:
print("\n3.2. Set training and model parameters:")
output = 'class'
predict = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']

#Build models
print("\n4. Build Models")

#Build Logistic Regression models

#Logistic Regression model 1: trained on whole data set
print("\n--Logistic Regression(1): Trained on whole data set")
s=time.time()
lr = LogisticRegression(max_iter=10000, fit_intercept=False, C=10000)
lr = classification_model(lr, NewData, predict, output)
predictions = lr.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(lr, open('LR_model.sav', 'wb'))

#Logistic Regression model 2: trained on Train set, tested on Test set
print("\n--Logistic Regression (2): With Train Test Split")
s=time.time()
lr2 = LogisticRegression(max_iter=10000, fit_intercept=False, C=10000)
X_train[predict]
lr = classification_model2(lr2, X_train, predict, y_train)
predictions = lr2.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(lr2, open('LR2_model.sav', 'wb'))

#Build K-Nearest Neighbors Models

#K-Nearest Neighbors Model (1)
print("\n--K-Nearest Neighbors (1)")
s=time.time()
knn = KNeighborsClassifier(weights='distance', n_neighbors=200)
knn = classification_model(knn, NewData, predict, output)
predictions = knn.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(knn, open('KNN_model.sav', 'wb'))

#K-Nearest Neighbors Model (2): trained on Train set, tested on Test set
print("\n--K-Nearest Neighbors (2): with Train Test Split")
s=time.time()
knn2 = KNeighborsClassifier(weights='distance', n_neighbors=150)
knn2 = classification_model2(knn2, X_train, predict, y_train)
predictions = knn2.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(knn2, open('KNN2_model.sav', 'wb'))

#Build Decision Tree Classifier models

#Decision Tree Classifier model (1)
print("\n--Decision Tree (1)")
s=time.time()
dtree = DecisionTreeClassifier(random_state=40, max_depth=20, max_leaf_nodes=100)
dtree = classification_model(dtree, NewData, predict, output)
predictions = dtree.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(dtree, open('DT_model.sav', 'wb'))

#Decision Tree Classifier model (2): trained on Train set, tested on Test set
print("\n--Decision Tree (2): with Train Test Split")
s=time.time()
dtree2 = DecisionTreeClassifier(random_state=40)#,max_features='sqrt')
dtree2 = classification_model2(dtree2, X_train, predict, y_train)
predictions = dtree2.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(dtree2, open('DT2_model.sav', 'wb'))

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

#Build Naive Bayes model

#Naive Bayes model (1) 
print("\n--Naive Bayes model (1)")
s=time.time()
nb= GaussianNB()
nb = classification_model(nb,NewData,predict,output)
predictions = nb.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(nb, open('NB_model.sav', 'wb'))

#Naive Bayes model (2) 
print("\n--Naive Bayes model (1): Using Train Test split")
s=time.time()
nb2= GaussianNB()
nb2 = classification_model2(nb2, X_train, predict, y_train)
predictions = nb2.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(nb2, open('NB2_model.sav', 'wb'))

#Build Surport Vector Machine models

#Surport Vector Model (1)
print("\n--Surportting vector machine (1)")
s=time.time()
svc = svm.SVC()
svc = classification_model(svc,NewData,predict,output)
predictions = svc.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions,y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(svc, open('SVM_model.sav', 'wb'))

#Support vector machine (2): using Train Test split
print("\n--Surportting vector machine (2): Using Train Test split")
s=time.time()
svc2 = svm.SVC()
X_train[predict]
svc2 = classification_model2(svc2, X_train, predict, y_train)
predictions = svc2.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(svc2, open('SVM_model.sav', 'wb'))

#Build ensemble learning models
estimators = [('lr',lr),('knn',knn),('tree',dtree)]

#Soft Voting (Logistic Regression + K-Nearest Neighbors + Decision Tree) (1)
print("\n--Soft voting (LR+KNN+DT) (1) ")
s=time.time()
soft_vote = VotingClassifier(estimators=estimators , voting= 'soft')
soft_vote = classification_model(soft_vote, NewData, predict, output)
predictions = soft_vote.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(soft_vote, open('SV_model.sav', 'wb'))

#Soft Voting (Logistic Regression + K-Nearest Neighbors + Decision Tree) (2): With Train-Test Split
print("\n--Soft Voting (LR+KNN+DT) (2): using Train Test Split ")
s=time.time()
soft_vote2 = VotingClassifier(estimators=estimators , voting= 'soft')
soft_vote2 = classification_model2(soft_vote2, X_train, predict, y_train)
predictions = soft_vote2.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(soft_vote2, open('SV2_model.sav', 'wb'))

#Hard Voting (Logistic Regression + K-Nearest Neighbors + Decision Tree) (1)
print("\n--Hard Voting")
s=time.time()
hard_vote = VotingClassifier(estimators=estimators , voting= 'hard')
hard_vote = classification_model(hard_vote, NewData, predict, output)
predictions = hard_vote.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(hard_vote, open('HV_model.sav', 'wb'))

#Hard Voting (Logistic Regression + K-Nearest Neighbors + Decision Tree) (2): With Train-Test Split
print("\n--Hard Voting (2): Using Train Test Split")
s=time.time()
hard_vote2 = VotingClassifier(estimators=estimators , voting= 'hard')
hard_vote2 = classification_model2(hard_vote2, X_train, predict, y_train)
predictions = hard_vote2.predict(X_test[predict])    
accuracy = metrics.accuracy_score(predictions,y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(hard_vote2, open('HV2_model.sav', 'wb'))

#Stacking using KNN as Meta (estimators = [('lr2',lr2),('knn',knn2),('tree',dtree2),('hard2',hard_vote2)]
s=time.time()
print("\n--Stacking (1) using KNN as Meta: Estimators = lr2, knn2, dtree2, hard_vote2")
meta = KNeighborsClassifier(weights='distance', n_neighbors=100)
stack = StackingClassifier(estimators = [('lr2',lr2),('knn',knn2),('tree',dtree2),('hard2',hard_vote2)])
stack = classification_model2(stack, X_train, predict, y_train) 
predictions = stack.predict(X_test[predict])
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(stack, open('Stack_model.sav', 'wb'))

#Stacking using Logistic Regression as meta (estimators = [('knn',knn),('tree',dtree),('soft',soft_vote),('rf',rf)])
print("\n--Stacking (2) using Logistic Regression as Meta: Estimators = knn, dtree, soft_vote, rf")
s=time.time()
meta = LogisticRegression(max_iter=10000, fit_intercept=False, C=10000)
stack2 = StackingClassifier(estimators = [('knn',knn),('tree',dtree),('soft',soft_vote),('rf',rf)], final_estimator=meta) 
stack2 = classification_model(stack2, NewData, predict, output)
predictions = stack2.predict(X_test[predict])
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on test data = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(stack2, open('Stack2_model.sav', 'wb'))

#Stacking using Logistic Regression as meta (estimators = [('knn2',knn2),('tree',dtree),('soft2',soft_vote2),('hard2',hard_vote2)])
print("\n--Stacking (3) using Logistic Regression as Meta: estimators = knn2, dtree, soft_vote2, hard_vote2")
s=time.time()
meta = LogisticRegression(max_iter=10000, fit_intercept=False, C=10000)
stack3 = StackingClassifier(estimators = [('knn2',knn2),('tree',dtree),('soft2',soft_vote2),
                                          ('hard2',hard_vote2)],final_estimator=meta) 
stack3 = classification_model2(stack3, X_train, predict, y_train)
predictions = stack3.predict(X_test[predict])
accuracy = metrics.accuracy_score(predictions, y_test)
print("Accuracy on Test Set = {}".format(accuracy))
print("Time = {}".format(time.time()-s))
pickle.dump(stack3, open('Stack3_model.sav', 'wb'))


