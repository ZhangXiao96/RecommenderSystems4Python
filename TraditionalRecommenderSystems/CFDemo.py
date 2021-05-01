import numpy as np
import pandas as pd
from TraditionalRecommenderSystems.CollaborativeFiltering.UserCF import UserCF
from TraditionalRecommenderSystems.CollaborativeFiltering.ItemCF import ItemCF
from lib.metric import cosine
from sklearn.model_selection import train_test_split
import os

# load dataset
data_root = os.path.join('..', 'Dataset', 'MovieLens', 'ml-latest-small')
rating_data_path = os.path.join(data_root, 'ratings.csv')
movie_data_path = os.path.join(data_root, 'movies.csv')
model = 'UserCF'

rating_data = pd.read_csv(rating_data_path)
movie_data = pd.read_csv(movie_data_path)
data = pd.merge(rating_data, movie_data, on='movieId')

unique_users = list(set(data['userId'].values))
unique_items = list(set(data['movieId'].values))

# split training set and test set
train_index, test_index = train_test_split(np.arange(len(data)), test_size=0.1, shuffle=True, random_state=42)
train_data = data.loc[train_index]
test_data = data.loc[test_index]

paired_train_data = list(zip(train_data['userId'].values, train_data['movieId'].values, train_data['rating'].values))
paired_test_data = list(zip(test_data['userId'].values, test_data['movieId'].values))
test_ground_truth = test_data['rating'].values

if model == 'UserCF':
    # using UserCF to see the predicted ratings of the movie
    CF = UserCF(paired_train_data, unique_users, unique_items, nb_similar_user=20,
                fillna="constant", value=0, similarity=cosine)
elif model == 'ItemCF':
    # using ItemCF to see the predicted ratings of the movie
    CF = ItemCF(paired_train_data, unique_users, unique_items, nb_similar_item=20,
                fillna="constant", value=0, similarity=cosine)
else:
    raise Exception('No model named {}!'.format(model))

# top 10 recommendation (id and score).
test_users, test_items = test_data['userId'].values, test_data['movieId'].values
test_unique_users = list(set(test_users))
id_recommendation, rating_recommendation = CF.recommend(test_unique_users, 10)
prediction = CF.predict_ratings(paired_test_data)

# calculate precision and recall
test_dict = {}
# Build ground truth
for i in range(len(test_data)):
    if test_users[i] not in test_dict:
        test_dict[test_users[i]] = {test_items[i]}
    else:
        test_dict[test_users[i]].add(test_items[i])

precision, recall = 0, 0
for i, user in enumerate(test_unique_users):
    pred_items = set(CF.indices_to_items(id_recommendation[i].ravel()))
    hit = pred_items & test_dict[user]
    precision += len(hit) / len(pred_items)
    recall += len(hit) / len(test_dict[user])

precision /= len(test_unique_users)
recall /= len(test_unique_users)

# using MSE to evaluate the performance
loss = np.mean((prediction-test_ground_truth)**2)
print("The test MSE loss of rating is {}.".format(loss))
print('precision={}, recall={}'.format(precision, recall))
