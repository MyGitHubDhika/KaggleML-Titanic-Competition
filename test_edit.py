import pandas as pd
import numpy as np

filepath = "test.csv"
test_data = pd.read_csv(filepath)

test_data['Age'] = test_data['Age'].fillna(round(test_data['Age'].mean()))
test_data['Fare'] = test_data['Fare'].fillna(round(test_data['Fare'].mean()))

test_data = test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

test_data['Sex'] = np.where(test_data['Sex'] == 'male', 1, 0)

test_data['Embarked'] = test_data['Embarked'].astype('category').cat.codes

#test_data.to_csv('adv_test.csv')