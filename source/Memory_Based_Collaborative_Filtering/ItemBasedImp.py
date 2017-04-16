import pandas as pd
import numpy as np
from scipy import spatial

# create user item ratings matrix

def get_user_ratings_matrix(data_file):
    # here we define column name in our data file
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    # loading data in our raw data file into pandas dataframe
    user_ratings = pd.read_csv(data_file, sep='\s+', names=r_cols)
    # removing timestamp column as it is not needed in this stage
    user_ratings.__delitem__('timestamp')
    # here we aggregate using a similar function like groupby
    # here we use mean function as the aggregate funtion. One important thing is reindexing. You can try with and
    # without reindexing. Using reindexing adds column for never rated movie.
    user_ratings_matrix = user_ratings.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=0,
                                                   aggfunc=np.mean).reindex(columns=np.arange(1, 1683), fill_value=0)
    return user_ratings_matrix

# create item similarity matrix
def get_item_similarity_matrix(user_item_ratings):
    transpose_dataframe = user_item_ratings.T
    number_of_items = len(transpose_dataframe.index)
    item_similarity_matrix = np.zeros(shape=(number_of_items, number_of_items))
    print item_similarity_matrix.shape

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