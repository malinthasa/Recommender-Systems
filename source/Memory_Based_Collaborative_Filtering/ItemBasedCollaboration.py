from ItemBasedImp import get_user_ratings_matrix
from ItemBasedImp import get_item_similarity_matrix
from ItemBasedImp import get_weights

train_file = '/home/malintha/projects/Recommender-Systems/test/resources/data/ml-100k/ua.base'
test_file = "/home/malintha/projects/Recommender-Systems/test/resources/data/ml-100k/ua.test"

user_ratings_matrix = get_user_ratings_matrix(test_file)
item_similarity_matrix = get_item_similarity_matrix(user_ratings_matrix)
print get_weights(0,0,user_ratings_matrix,item_similarity_matrix)