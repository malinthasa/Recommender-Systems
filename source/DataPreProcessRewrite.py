import pandas as pd
import numpy as np
from random import randint
k= 10
number_of_users = 943
number_of_movies = 1682
def get_user_ratings_matrix(data_file):
    # here we define column name in our data file
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    # loading data in our raw data file into pandas dataframe
    user_ratings = pd.read_csv(data_file, sep='\s+', names=r_cols)
    # removing timestamp column as it is not needed in this stage
    user_ratings.__delitem__('timestamp')
    # here we aggregate using a similar function like groupby
    # here we use mean function as the aggregate funtion. One important thing is reindexing. You can try with and without
    # reindexing. Using reindexing adds column for never rated movie.
    user_ratings_matrix = user_ratings.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=0,
                                                   aggfunc=np.mean).reindex(columns=np.arange(1, number_of_movies + 1), fill_value=0)
    return user_ratings_matrix

def get_train_test_data(user_ratings):
    # print user_ratings
    test_data = np.zeros(shape=(number_of_users))
    for index, row in user_ratings.iterrows():
        selected_user_for_testing = np.array(row.argsort()[::-1][:k])[randint(0, k-1)]
        test_data[index - 1] = selected_user_for_testing
        row.iloc[selected_user_for_testing] = 0
    # print test_data + 1
    return user_ratings, test_data

