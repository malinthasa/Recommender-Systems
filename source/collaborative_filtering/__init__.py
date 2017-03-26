import pandas as pd
import numpy as np
from CFImplementation import data_preprocess

test_file = "/home/malintha/projects/Recommender-Systems/test/resources/data/ml-100k/u1.test"
# here we define column name in our data file
r_cols = ['user_id', 'movie_id', 'rating','timestamp']
# loading data in our raw data file into pandas dataframe
user_ratings = pd.read_csv(test_file, sep='\s+', names=r_cols)
# removing timestamp column as it is not needed in this stage
user_ratings.__delitem__('timestamp')
# here we aggregate using a similar function like groupby
# here we use mean function as the aggregate funtion. One important thing is reindexing. You can try with and without
# reindexing. Using reindexing adds column for never rated movie.
user_ratings_matrix = user_ratings.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=0,
                                                   aggfunc=np.mean).reindex(columns=np.arange(1,710), fill_value=0)
user_count = 10
evaluate = 0
for user_id in range(user_count):
    recommended_movies = data_preprocess('/home/malintha/projects/Recommender-Systems/test/resources/data/ml-100k/u1.base', user_id + 1, 15)
    temp1 = np.where(user_ratings_matrix.ix[user_id + 1].as_matrix() != 0)[0]
    if len(np.intersect1d(temp1, recommended_movies - 1)) > 0:
        evaluate += 1

print(evaluate * 100 / 10 )
