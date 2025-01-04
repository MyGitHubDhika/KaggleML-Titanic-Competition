import pandas as pd

titanic_file_path = 'C:/Users/apria/gitProjects/Python/KaggleLearn/Comp_Titanic/Titanic_Dataset/train.csv'
titanic_data = pd.read_csv(titanic_file_path, index_col='PassengerId')

print(titanic_data['Age'].describe())