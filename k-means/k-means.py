import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

train = pd.read_csv('./data/dataset/iris_train.csv')
test = pd.read_csv('./data/dataset/iris_test.csv')

X_train = train.drop(columns=['species']).to_numpy()
y_train = train['species'].to_numpy()
X_test = test.drop(columns=['species']).to_numpy()

kmeans = KMeans(n_clusters=3).fit(X_train, y_train)
y_pred = kmeans.predict(X_test).tolist()

with open('k-means/predict.txt', 'w') as f:
    for el in y_pred:
        f.write(f"{el} \n")