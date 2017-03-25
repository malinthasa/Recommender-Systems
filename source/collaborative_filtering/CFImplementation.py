import pandas as pd
import numpy as np
from scipy import spatial

def get_user_similarity(user_ratings):
    number_of_users = len(user_ratings.index)
    user_similarity_matrix = np.zeros(shape=(number_of_users, number_of_users))
    most_similar_users = np.zeros(shape=(number_of_users, number_of_users - 1))
    for index, row in user_ratings.iterrows():
        for index_internal, row_internal in user_ratings.iterrows():
            user_similarity_matrix[index - 1][index_internal - 1] = 1 - spatial.distance.cosine(row.as_matrix(),
                                                                                                row_internal.as_matrix())
        most_similar_users[index - 1] = np.array(user_similarity_matrix[index - 1]).argsort()[::-1][1:]
    return most_similar_users

def data_preprocess(data_file):
    # here we define column name in our data file
    r_cols = ['user_id', 'movie_id', 'rating','timestamp']
    # loading data in our raw data file into pandas dataframe
    user_ratings = pd.read_csv(data_file, sep='\s+', names=r_cols)
    # removing timestamp column as it is not needed in this stage
    user_ratings.__delitem__('timestamp')
    # here we aggregate using a similar function like groupby
    # here we use mean function as the aggregate funtion. One important thing is reindexing. You can try with and without
    # reindexing. Using reindexing adds column for never rated movie.
    user_ratings_matrix = user_ratings.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=0,
                                                   aggfunc=np.mean).reindex(columns=np.arange(1,6), fill_value=0)
    # now we have user-movie ratings matrix
    # Creates a list containing 5 lists, each of 8 items, all set to 0
    print get_user_similarity(user_ratings_matrix)

    #go through each user and find the most similar user for that user
    return user_ratings_matrix