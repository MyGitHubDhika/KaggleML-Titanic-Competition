import pandas as pd

train_filepath = 'adv_train.csv'
train_data = pd.read_csv(train_filepath)

train_y = train_data.Survived

train_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_X = train_data[train_features]

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=1)
model.fit(train_X, train_y)

test_filepath = 'adv_test.csv'
test_data = pd.read_csv(test_filepath)

test_features = train_features
test_X = test_data[test_features]

prediction = pd.DataFrame({
	'PassengerId': test_data['PassengerId'],
	'Survived': model.predict(test_X)
})

prediction.to_csv('titanic_submission.csv', index=False)