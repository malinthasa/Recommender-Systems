from CFImplementation import data_preprocess

user_count = 10
for user_id in range(user_count):
    data_preprocess('/home/malintha/projects/Recommender-Systems/test/resources/data/ml-100k/u1.base', user_id + 1, 10)
