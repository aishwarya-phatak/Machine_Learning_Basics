import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

iris_dataset = load_iris()
print("-----------Iris dataset:-------------")
print(iris_dataset)

print("-----------keys from dataset:-----------")
print(iris_dataset.keys())
iris_data = iris_dataset.data
print("-----------data key from dataset:-----------")
print(iris_data)
print(iris_data.shape)

model = KMeans(n_clusters=3,random_state=42)
model.fit(iris_data)

print("Model labels : ", model.labels_)

#silhouette score ranges between -1 to +1
s_score = silhouette_score(iris_data,model.labels_)
print("Silhouette score for iris dataset:", s_score)

plt.scatter(iris_data[0:150,2], iris_data[0:150,3], c = model.labels_)
plt.title("K-means clustering for iris dataset")
plt.xlabel("Petal Length in cm")
plt.ylabel("Petal Width in cm")
plt.show()