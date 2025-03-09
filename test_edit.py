import pandas as pd
import numpy as np

filepath = "C:/Users/apria/Project/Python/TitanicML_Competition/Titanic_Dataset/test.csv"
data = pd.read_csv(filepath)

data['Age'] = data['Age'].fillna(round(data['Age'].mean()))
data['Fare'] = data['Fare'].fillna(round(data['Fare'].mean()))

data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

data['Sex'] = np.where(data['Sex'] == 'male', 1, 0)

data['Embarked'] = data['Embarked'].astype('category').cat.codes

data.to_csv('C:/Users/apria/Project/Python/TitanicML_Competition/Titanic_Dataset/adv_test.csv')