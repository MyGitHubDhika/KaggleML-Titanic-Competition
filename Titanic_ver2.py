import pandas as pd

filepath = 'C:/Users/apria/Project/Python/TitanicML_Competition/Titanic_Dataset/adv_train.csv'
train_data = pd.read_csv(filepath)

y = train_data.Survived

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_data[features]

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=1)
model.fit(X, y)
prediction = model.predict(X)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y, prediction)