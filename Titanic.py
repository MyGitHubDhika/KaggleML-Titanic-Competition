import pandas as pd
import numpy as np

titanic_file_path = 'C:/Users/apria/gitProjects/Kaggle_ML/TitanicCompetition/titanic/train.csv'
titanic_data = pd.read_csv(titanic_file_path, index_col='PassengerId')
titanic_data = titanic_data.drop('Cabin', axis=1)
titanic_data = titanic_data.dropna(axis=0)
titanic_data['Sex'] = np.where(titanic_data['Sex'] == 'male', 1, 0)

y = titanic_data.Survived
titanic_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = titanic_data[titanic_features]

from sklearn.ensemble import RandomForestClassifier

titanic_model = RandomForestClassifier(random_state=1)
titanic_model.fit(X, y)

test_file_path = 'C:/Users/apria/gitProjects/Kaggle_ML/TitanicCompetition/titanic/test.csv'
test_data = pd.read_csv(test_file_path)
test_data = test_data.drop('Cabin', axis=1)
test_data = test_data.dropna(axis=0)
test_data['Sex'] = np.where(test_data['Sex'] == 'male', 1, 0)

test_X = test_data[titanic_features]

prediction = titanic_model.predict(test_X)