# Used to load the dataset and preprocess it for the experiment

import pandas as pd

train_filepath = 'train.csv' # Path to the train.csv file
train_data = pd.read_csv(train_filepath, index_col='PassengerId')
train_data = train_data.drop('Cabin', axis=1)
train_data = train_data.fillna(round(train_data['Age'].mean()))

