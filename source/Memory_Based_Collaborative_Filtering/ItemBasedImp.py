import pandas as pd
import numpy as np
from scipy import spatial

# create item similarity matrix
def get_item_similarity_matrix(user_item_ratings):
    transpose_dataframe = user_item_ratings.T
    number_of_items = len(transpose_dataframe.index)
    item_similarity_matrix = np.zeros(shape=(number_of_items, number_of_items))

    # Iterating through user_ratings matrix
    # print user_item_ratings
    for index, row in transpose_dataframe.iterrows():
        # For each user we calculate similarity to each other users
        for index_internal, row_internal in transpose_dataframe.iterrows():
            # Here we calculate the similarity using cosine similarity method
            distance = spatial.distance.cosine(row.as_matrix(),row_internal.as_matrix())
            if np.isnan(distance):
                distance = 1
            item_similarity_matrix[index - 1][index_internal - 1] = 1 - distance
    return item_similarity_matrix

# create ratings prediction based on weights
def get_predicted_ratings(user_index, item_index, user_ratings, item_similarity):
    item_similarities = item_similarity[item_index]
    sum_of_similarities = np.sum(item_similarities)
    predicted_rating = np.dot(user_ratings.iloc[user_index], item_similarities) / sum_of_similarities
    return int(round(predicted_rating))