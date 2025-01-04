import pandas as pd
import numpy as np

train_file_path = 'C:/Users/apria/gitProjects/Kaggle_ML/TitanicCompetition/titanic/train.csv'
train_data = pd.read_csv(train_file_path, index_col='PassengerId')
train_data = train_data.drop('Cabin', axis=1)
train_data = train_data.dropna(axis=0)
train_data['Sex'] = np.where(train_data['Sex'] == 'male', 1, 0)

y = train_data.Survived
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = train_data[features]

from sklearn.ensemble import RandomForestClassifier

train_model = RandomForestClassifier(random_state=1)
train_model.fit(X, y)

test_file_path = 'C:/Users/apria/gitProjects/Kaggle_ML/TitanicCompetition/titanic/test.csv'
test_data = pd.read_csv(test_file_path)
test_data = test_data.drop('Cabin', axis=1)
test_data = test_data.dropna(axis=0)
test_data['Sex'] = np.where(test_data['Sex'] == 'male', 1, 0)

test_X = test_data[features]

prediction = train_model.predict(test_X)