import pandas as pd
import numpy as np

train_path = "C:/Users/apria/Project/Programming_Main/Python/TitanicML_Competition/Titanic_Dataset/train.csv"
train_data = pd.read_csv(train_path)

print(train_data['Age'].mean())
train_data['Age'] = train_data['Age'].fillna(round(train_data['Age'].mean()))

train_data = train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

train_data['Sex'] = np.where(train_data['Sex'] == 'male', 1, 0)

train_data = train_data.dropna()
train_data['Embarked'] = train_data['Embarked'].astype('category').cat.codes

train_data.to_csv('C:/Users/apria/Project/Programming_Main/Python/TitanicML_Competition/Titanic_Dataset/adv_train.csv')