import numpy as np
from UserBasedImp import data_pre_process
from utils import get_user_ratings_matrix

train_file = '/home/malintha/projects/Recommender-Systems/test/resources/data/ml-100k/ua.base'
test_file = "/home/malintha/projects/Recommender-Systems/test/resources/data/ml-100k/ua.test"

test_user_ratings_matrix = get_user_ratings_matrix(test_file)
original_user_ratings_matrix = get_user_ratings_matrix(train_file)
user_count = 943
evaluate = 0
k = 10

for user_id in range(user_count):
    recommended_movies = data_pre_process(user_id + 1, k, original_user_ratings_matrix)
    print user_id
    temp1 = np.where(test_user_ratings_matrix.ix[user_id + 1].as_matrix() != 0)[0]
    if len(np.intersect1d(temp1, recommended_movies - 1)) > 0:
        evaluate += 1

print(evaluate * 100 / user_count )