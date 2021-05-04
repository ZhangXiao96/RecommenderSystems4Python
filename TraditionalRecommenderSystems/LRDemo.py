"""
Note: The key to Logistic Regression is the construction of feature vectors,
which directly decides the performance of the model. Since the construction
of feature vectors varies according to datasets, here we only provide a simple
example of how to use the LR in sklearn. We set the label as "like" when rating>=4
and "dislike" when rating<4.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from TraditionalRecommenderSystems.LogisticRegression.LogisticRegression import BaseLogisticRegression
import os

# load dataset
data_root = os.path.join('..', 'Dataset', 'MovieLens', 'ml-latest-small')
rating_data_path = os.path.join(data_root, 'ratings.csv')
movie_data_path = os.path.join(data_root, 'movies.csv')

rating_data = pd.read_csv(rating_data_path)
movie_data = pd.read_csv(movie_data_path)
data = pd.merge(rating_data, movie_data, on='movieId')

unique_users = list(set(data['userId'].values))
unique_items = list(set(data['movieId'].values))
unique_genres = set()
for genres in movie_data['genres']:
    unique_genres |= set(genres.split('|'))
# split training set and test set
train_index, test_index = train_test_split(np.arange(len(data)), test_size=0.1, shuffle=True, random_state=42)
train_data = data.loc[train_index]
test_data = data.loc[test_index]

# feature extraction
# build movie description
genres_encoder = {}
for i, genres in enumerate(unique_genres):
    genres_encoder[genres] = i
one_hot = np.eye(len(genres_encoder))
movie_description = {}
for movie_id, movie_genres in zip(movie_data['movieId'], movie_data['genres']):
    labels = np.array([genres_encoder[_] for _ in set(movie_genres.split('|'))])
    movie_description[movie_id] = np.sum(one_hot[labels, :], axis=0)

# build user Description
# rate of different categories of movies, and the rate of watched movie.
user_description = {}
for user in unique_users:
    user_description[user] = np.zeros(shape=[len(unique_genres)+1])
for user, item in zip(train_data['userId'], train_data['movieId']):
    user_description[user][:-1] += movie_description[item]
    user_description[user][-1] += 1
for user in unique_users:
    user_description[user][:-1] /= user_description[user][-1]
    user_description[user][-1] /= len(movie_description)

# build Interaction Description

test_y = np.where(test_data['rating'].values >= 4, 1, 0)
train_y = np.where(train_data['rating'].values >= 4, 1, 0)

paired_train_data = list(zip(train_data['userId'].values, train_data['movieId'].values, train_y))
paired_test_data = list(zip(test_data['userId'].values, test_data['movieId'].values))
test_ground_truth = test_y


class_weight = {1: np.mean(train_y), 0: 1-np.mean(train_y)}
LR = BaseLogisticRegression(paired_train_data, user_description, movie_description, C=1., class_weight=class_weight)

LR.train()
print(LR.eval(paired_test_data, test_ground_truth))
#
# MF = MatrixFactorization(paired_train_data, user_list=unique_users, item_list=unique_items, nb_factor=80, lr=1e-3,
#                          weight_decay=1e-6, batch_size=64, drop_rate=0.2, pro_process=None)
#
# train_loss, test_loss = MF.train(epochs=80, test_data=list(zip(test_data['userId'].values, test_data['movieId'].values,
#                                                                test_data['rating'].values)), test_epoch_step=1)
#
# # top 10 recommendation (id and score).
# test_users, test_items = test_data['userId'].values, test_data['movieId'].values
# test_unique_users = list(set(test_users))
# id_recommendation, rating_recommendation = MF.recommend(test_unique_users, 10)
# prediction = MF.predict_ratings(paired_test_data)
#
# a = MF.model.get_rating_matrix().data.cpu().numpy()
# b = MF.history_rating_matrix.values
#
# # calculate precision and recall
# test_dict = {}
# # Build ground truth
# for i in range(len(test_data)):
#     if test_users[i] not in test_dict:
#         test_dict[test_users[i]] = {test_items[i]}
#     else:
#         test_dict[test_users[i]].add(test_items[i])
#
# precision, recall = 0, 0
# for i, user in enumerate(test_unique_users):
#     pred_items = set(MF.indices_to_items(id_recommendation[i].ravel()))
#     hit = pred_items & test_dict[user]
#     precision += len(hit) / len(pred_items)
#     recall += len(hit) / len(test_dict[user])
#
# precision /= len(test_unique_users)
# recall /= len(test_unique_users)
#
# # using MSE to evaluate the performance
# loss = np.mean((prediction-test_ground_truth)**2)
# print("The test MSE loss of rating is {}.".format(loss))
# print('precision={}, recall={}'.format(precision, recall))
