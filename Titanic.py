import pandas as pd
import numpy as np

train_filepath = 'C:/Users/apria/gitProjects/Python/KaggleLearn/Comp_Titanic/Titanic_Dataset/train.csv'
train_data = pd.read_csv(train_filepath, index_col='PassengerId')
train_data = train_data.drop('Cabin', axis=1)
train_data = train_data.fillna(round(train_data['Age'].mean()))
train_data['Sex'] = np.where(train_data['Sex'] == 'male', 1, 0)

y = train_data.Survived
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = train_data[features]

from sklearn.ensemble import RandomForestClassifier

train_model = RandomForestClassifier(random_state=1)
train_model.fit(X, y)

test_filepath = 'C:/Users/apria/gitProjects/Python/KaggleLearn/Comp_Titanic/Titanic_Dataset/test.csv'
test_data = pd.read_csv(test_filepath)
test_data = test_data.drop('Cabin', axis=1)
train_data = train_data.fillna(round(train_data['Age'].mean()))
test_data['Sex'] = np.where(test_data['Sex'] == 'male', 1, 0)

test_X = test_data[features]

prediction = train_model.predict(test_X)

result_dict = {
    "PassengerId": test_data['PassengerId'].tolist(),
    "Survived": prediction.tolist()
    }

result_data = pd.DataFrame(result_dict)
result_data.to_csv('C:/Users/apria/gitProjects/Python/KaggleLearn/Comp_Titanic/result.csv', index=False)