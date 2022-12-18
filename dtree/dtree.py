import pandas as pd

from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv('./data/dataset/iris_train.csv')
X_train = train.drop(columns=['species']).to_numpy()
y_train = train['species'].to_numpy()

test = pd.read_csv('./data/dataset/iris_test.csv')
X_test = test.drop(columns=['species']).to_numpy()

tree = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = tree.predict(X_test).tolist()

with open('dtree/predict.txt', 'w') as f:
    for el in y_pred:
        f.write(f"{el} \n")