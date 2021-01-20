# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 21:23:24 2021

@author: steph
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha=0.8, cmap=mglearn.cm3)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)#on fournit au modèle les données d'apprentissage

X_train[0]

prediction = knn.predict(X_test)
print(f"Prediction: {prediction}")
print(f"Prediction target name: {iris_dataset['target_names'][prediction]}")

print(f"Score: {knn.score(X_test, y_test)}")




#print(f'Iris dataset keys: {iris_dataset.keys()})')
#print(iris_dataset['target_names'])

