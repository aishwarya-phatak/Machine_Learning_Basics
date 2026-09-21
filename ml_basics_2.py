import pandas as pd

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

#important
df = pd.get_dummies(data_frame)
print(type(df))
print(df)