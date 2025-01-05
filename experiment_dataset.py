import pandas as pd

train_filepath = 'C:/Users/apria/gitProjects/Python/KaggleLearn/Comp_Titanic/Titanic_Dataset/train.csv'
train_data = pd.read_csv(train_filepath, index_col='PassengerId')
train_data = train_data.drop('Cabin', axis=1)
train_data = train_data.fillna(round(train_data['Age'].mean()))

