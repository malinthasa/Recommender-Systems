import numpy as np
from LapLRS import LapLRSFunc
from Evaluating import evaluate
from DataPreProcess import data_pre_processor
from DataPreProcessRewrite import get_user_ratings_matrix
from DataPreProcessRewrite import get_train_test_data

# # Obtain training set and the test set
# pre_processed_data = data_pre_processor('/home/malintha/projects/Recommender-Systems/test/resources/data/ml-100k/test', 1)
#
# # obtain W
# weight_matrix = LapLRSFunc(np.asmatrix(pre_processed_data[0]))
#
# # obtain HR and ARHR
# HR, ARHR = evaluate(weight_matrix, np.asmatrix(pre_processed_data[0]),  pre_processed_data[1])
#
# print("HR is :%s",HR)
# print("ARHR is :%s",ARHR)

train_file = '/home/malintha/projects/Recommender-Systems/test/resources/data/ml-100k/u.data'
original_user_ratings_matrix = get_user_ratings_matrix(train_file)

trainnig_set, test_set = get_train_test_data(original_user_ratings_matrix)
# print test_set + 1
weight_matrix = LapLRSFunc(np.asmatrix(trainnig_set), test_set)

# HR, ARHR = evaluate(weight_matrix, np.asmatrix(trainnig_set),  test_set)

# print HR