#!/usr/bin/python
from __future__ import division
from pandas import *
import math

number_of_users = 943
number_of_movies = 1682
selected_recommendations = 10


def evaluate(weight, user_item_ratings, ratings_test):

    # Calculating predicted ratings matrix
    predicted = user_item_ratings * weight

    # Selecting top-N recommendations list
    # N=10
    top_n_ratings = np.zeros(shape=(number_of_users, selected_recommendations))
    temp = np.squeeze(np.asarray(predicted))

    # Removing already rated movied from the predicted set
    # Actually giving them minus infinity rating value, so that those would not be selected to top-N

    for (b, m), value in np.ndenumerate(temp):
        # user_rated_list = np.where(np.array(user_item_ratings[b]) != 0)[0]
        user_rated_list = np.where(np.asarray(user_item_ratings[b])[0] != 0)[0]
        for i in user_rated_list:
            temp[b][i] = float("-inf")
        # then select top-N predicted ratings
        top_n_ratings[b] = np.argsort(temp[b])[-selected_recommendations:][::-1]

    # Calculating HR and ARHR
    # test_ratings = {k: g['movie_id'].tolist() for k, g in ratings_test.groupby('user_id')}
    user_count = len(ratings_test)
    hit_count = 0
    ARHR_count = 0

    temp_count = 0
    for row in top_n_ratings:
       if(ratings_test[temp_count] in row):
           hit_count +=1
       temp_count += 1

    return hit_count / user_count, ARHR_count / user_count
