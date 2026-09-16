import pandas as pd
import sklearn as skl
import sklearn.datasets as datasets

iris_dataset = datasets.load_iris()
print(iris_dataset.keys())
print(type(iris_dataset.keys()))

print("\n=========data===========\n")
iris_data = iris_dataset.data
print(iris_data)
print(type(iris_data))

print("\n=========target===========\n")
print(iris_dataset.target)
print(type(iris_dataset.target))

print("\n============frame=========\n")
print(iris_dataset.frame)
print(type(iris_dataset.frame))

print("\n=========feature names===========\n")
print(iris_dataset.feature_names)
print(type(iris_dataset.feature_names))

print("\n===========target names===============\n")
print(iris_dataset.target_names)
print(type(iris_dataset.target_names))

print("\n=============DESCR===============\n")
print(iris_dataset.DESCR)
print(type(iris_dataset.DESCR))

print("\n=============filename===============\n")
print(iris_dataset.filename)
print(type(iris_dataset.filename))

print("\n=============data module===============\n")
print(iris_dataset.data_module)
print(type(iris_dataset.data_module))






# 'data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'


