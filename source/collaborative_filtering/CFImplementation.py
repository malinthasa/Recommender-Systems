import pandas as pd
import numpy as np
from scipy import spatial

def get_top_n_recommendations(user_ratings_matrix, current_user):
    closest_user = get_most_similar_users(user_ratings_matrix, current_user)[0]
    closest_neighbours_ratings = user_ratings_matrix.ix[closest_user].as_matrix()
    current_users_unwathed_list = np.where(user_ratings_matrix.ix[current_user].as_matrix() == 0)[0]
    closest_neighbours_ratings_for_unwatched_movies = [closest_neighbours_ratings[index] for index in
                                                       current_users_unwathed_list]
    those_ratings_in_decending_order = np.array(closest_neighbours_ratings_for_unwatched_movies).argsort()[::-1][:]
    return [current_users_unwathed_list[a] for a in those_ratings_in_decending_order]

def get_most_similar_users(user_ratings, user_id):
    number_of_users = len(user_ratings.index)
    user_similarity_matrix = np.zeros(shape=(number_of_users, number_of_users))
    most_similar_users = np.zeros(shape=(number_of_users, number_of_users - 1))

    for index, row in user_ratings.iterrows():
        for index_internal, row_internal in user_ratings.iterrows():
            user_similarity_matrix[index - 1][index_internal - 1] = 1 - spatial.distance.cosine(row.as_matrix(),
                                                                                                row_internal.as_matrix())
        most_similar_users[index - 1] = np.array(user_similarity_matrix[index - 1]).argsort()[::-1][1:]
    return most_similar_users[user_id - 1] + 1

def data_preprocess(data_file,user, n):
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
                                                   aggfunc=np.mean).reindex(columns=np.arange(1,710), fill_value=0)
    return np.asarray(get_top_n_recommendations(user_ratings_matrix, user)[:n]) + 1