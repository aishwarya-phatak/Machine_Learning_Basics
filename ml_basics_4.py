import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

#random forest regressor

#convert csv to dataframe format
df = pd.read_csv('dataset/Housing.csv')

#checking on dataframe
#head
print("======================")
print(df.head())

#info
print('----------------------')
print(df.info())

#DESCRIBE
print('----------------------')
print(df.describe())

#null check
#if the data is null in any column then only use this, here it is used for your understanding
print(df.isnull())
print(df.isnull().sum())

#encoding
print(pd.get_dummies(df))