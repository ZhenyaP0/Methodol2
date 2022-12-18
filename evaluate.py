import pandas as pd

from sklearn.metrics import accuracy_score

y_test = pd.read_csv('data/dataset/iris_test.csv')['species'].to_numpy().tolist()

kmeans_pred = list()
with open('k-means/predict.txt', 'r') as f:
    for line in f:
        kmeans_pred.append(int(line))

#dtree_pred = list()
#with open('dtree/predict.txt', 'r') as f:
 #   for line in f:
  #      dtree_pred.append(int(line))

kmeans_report = accuracy_score(y_test, kmeans_pred)
#dtree_report = accuracy_score(y_test, dtree_pred)

with open('metrics.txt', 'w') as f:
 #   f.write(f"dtree acc: {dtree_report}")
  #  f.write("\n")
    f.write(f"kmeans acc: {kmeans_report}")