#Import packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import pickle

#Load data from .csv file

data = pd.read_csv("C:\Morphy\BankChurners\BankChurnersV2.csv")

print("1. Information of Raw Data:")
print("1.1. Data columns: ")
#print(data.head())
print(data.columns)
print("1.2. Data statistaics: ")
print(data.info())
print("1.3. Null data: ")
print(data.isnull().sum())

#Feature preprocessing
print("\n2. Feature preprocessing:")

data['Attrition_Flag'].replace({"Existing Customer":0,"Attrited Customer":1},inplace=True)
data['Gender'].replace({'F':0, 'M':1}, inplace=True)
data['Education_Level'].replace({'Graduate':0, 'High School':1, 'Uneducated':2, 'College':3, 'Unknown':4, 'Post-Graduate':5, 'Doctorate':6}, inplace=True)
data['Marital_Status'].replace({'Married':0, 'Single':1, 'Divorced':2, 'Unknown':3}, inplace=True)
data['Income_Category'].replace({'Less than $40K':0, '$40K - $60K':1, '$60K - $80K':2, '$80K - $120K':3, '$120K +':4, 'Unknown':5}, inplace=True)
data['Card_Category'].replace({'Blue':0, 'Silver':1, 'Gold':2, 'Platinum':3}, inplace=True)

print("2.1. Data statistaics after encoding: ")
print(data.info())
