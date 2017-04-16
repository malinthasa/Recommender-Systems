from ItemBasedImp import get_item_similarity_matrix
from ItemBasedImp import get_predicted_ratings
from utils import get_user_ratings_matrix

train_file = '/home/malintha/projects/Recommender-Systems/test/resources/data/ml-100k/test'
test_file = "/home/malintha/projects/Recommender-Systems/test/resources/data/ml-100k/ua.test"

user_ratings_matrix = get_user_ratings_matrix(train_file)
item_similarity_matrix = get_item_similarity_matrix(user_ratings_matrix)
print get_predicted_ratings(0, 7, user_ratings_matrix, item_similarity_matrix)