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
print(df.dropna())                  #drop --> removing missing or null values

pd.set_option('display.max_columns', None)      #to display all columns

#encoding
encoded_dataset = pd.get_dummies(df)
print(encoded_dataset)

x = encoded_dataset.drop(columns=['price'])
y = encoded_dataset['price']

X_train, X_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    train_size=0.8,
                                                    random_state=42)

#random forest regressor
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

#prediction
y_prediction = model.predict(X_test)

#metrics - evaluation of performance
mse = mean_squared_error(y_test, y_prediction)
print(f"mean square error is : {mse}")

mae = mean_absolute_error(y_test, y_prediction)
print(f"mean absolute error is : {mae}")

r2 = r2_score(y_test, y_prediction)
print(f"r2 score is : {r2}")