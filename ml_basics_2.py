import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

#/Users/vishaljagtap/Desktop/PCP_Python_Basics/Machine_Learning_Basics/dataset/Housing.csv  --> absolute path
#dataset/Housing.csv     --> path from content root

data_frame = pd.read_csv('dataset/Housing.csv')
print(type(data_frame))
print(data_frame)

print("-------------head method--------")
df_head = data_frame.head()
print(type(df_head))
print(df_head)

print("-------------info method--------")
df_info = data_frame.info()
print(type(df_info))
print(df_info)

print("-------------describe method--------")
df_describe = data_frame.describe()
print(type(df_describe))
print(df_describe)

df_dropna = data_frame.dropna()
print(type(df_dropna))
print(df_dropna)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 5)

#important
df = pd.get_dummies(data_frame)
print(type(df))
print(df)

x = df.drop(columns=['price'])
y = df['price']

X_train,X_test,y_train, y_test = train_test_split(x,
                                                  y,
                                                  test_size=0.2,
                                                  train_size=0.8)
model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

#comparison of testing and prediction
mse = mean_squared_error(y_test,y_pred)
print(f"mse : {mse}")

rmse = np.sqrt(mse)
print(f"rmse : {rmse}")

r2_score = r2_score(y_test,y_pred)
print(f"r2_score : {r2_score}")