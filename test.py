import pandas as pd


df=pd.read_csv('training.csv')
shuffled_ratings = df.sample(frac=1., random_state=1446557)
shuffled_ratings.to_csv('training1.csv')