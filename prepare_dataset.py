import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/evgeniapolezaeva/Methodol/data/iris.csv')
df['species'] = LabelEncoder().fit_transform(df['species'])

train, test = train_test_split(df, train_size=0.80)

train.to_csv('data/dataset/iris_train.csv')
test.to_csv('data/dataset/iris_test.csv')

print(df)
