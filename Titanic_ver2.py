import pandas as pd

train_filepath = 'C:/Users/apria/Project/Python/TitanicML_Competition/Titanic_Dataset/adv_train.csv'
train_data = pd.read_csv(train_filepath)

train_y = train_data.Survived

train_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_X = train_data[train_features]

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=1)
model.fit(train_X, train_y)

'''
prediction = model.predict(train_X)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y, prediction)
'''

test_filepath = 'C:/Users/apria/Project/Python/TitanicML_Competition/Titanic_Dataset/adv_test.csv'
test_data = pd.read_csv(test_filepath)

test_features = train_features
test_X = test_data[test_features]

prediction = pd.DataFrame({
	'PassengerId': test_data['PassengerId'],
	'Survived': model.predict(test_X)
})

prediction.to_csv('C:/Users/apria/Project/Python/TitanicML_Competition/Titanic_Dataset/titanic_submission.csv', index=False)