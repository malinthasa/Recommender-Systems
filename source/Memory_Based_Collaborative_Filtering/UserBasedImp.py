import pandas as pd
import numpy as np
from scipy import spatial

# Below function returns top-N recommendations for a given user
def get_top_n_recommendations(user_ratings_matrix, current_user):
    closest_user = get_most_similar_users(user_ratings_matrix, current_user)[0]
    closest_neighbours_ratings = user_ratings_matrix.ix[closest_user].as_matrix()
    current_users_unwathed_list = np.where(user_ratings_matrix.ix[current_user].as_matrix() == 0)[0]
    closest_neighbours_ratings_for_unwatched_movies = [closest_neighbours_ratings[index] for index in
                                                       current_users_unwathed_list]
    those_ratings_in_decending_order = np.array(closest_neighbours_ratings_for_unwatched_movies).argsort()[::-1][:]
    return [current_users_unwathed_list[a] for a in those_ratings_in_decending_order]


def get_most_similar_users(user_ratings, user_id):
    # Get number of users by getting the length of a single raw of the user-ratings matrix
    number_of_users = len(user_ratings.index)
    # Creating 2d zeros matrix | dimensions = number_of_users X number_of_users
    user_similarity_matrix = np.zeros(shape=(number_of_users, number_of_users))
    # Creating 2d zero matrix for most similar user. So its length should be number_of_users X number_of_users
    most_similar_users = np.zeros(shape=(number_of_users, number_of_users - 1))
    # Iterating through user_ratings matrix
    for index, row in user_ratings.iterrows():
        # For each user we calculate similarity to each other users
        for index_internal, row_internal in user_ratings.iterrows():
            # Here we calculate the similarity using cosine similarity method
            user_similarity_matrix[index - 1][index_internal - 1] = 1 - spatial.distance.cosine(row.as_matrix(),
                                                                                                row_internal.as_matrix())
            # We prepare calculated similarities in descending order and store indexes of most similarity calculations
        most_similar_users[index - 1] = np.array(user_similarity_matrix[index - 1]).argsort()[::-1][1:]
    # Getting the list of most similar users
    return most_similar_users[user_id - 1] + 1


def data_pre_process(user, n, user_ratings_matrix):
    return np.asarray(get_top_n_recommendations(user_ratings_matrix, user)[:n]) + 1
