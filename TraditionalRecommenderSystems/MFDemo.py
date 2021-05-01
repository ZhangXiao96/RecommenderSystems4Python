import numpy as np
import pandas as pd
from TraditionalRecommenderSystems.MatrixFactorization.MatrixFactorization import MatrixFactorization
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
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

# split training set and test set
train_index, test_index = train_test_split(np.arange(len(data)), test_size=0.1, shuffle=True, random_state=42)
train_data = data.loc[train_index]
test_data = data.loc[test_index]

paired_train_data = list(zip(train_data['userId'].values, train_data['movieId'].values, train_data['rating'].values))
paired_test_data = list(zip(test_data['userId'].values, test_data['movieId'].values))
test_ground_truth = test_data['rating'].values

# using MF to see the predicted ratings of the movie

#
# class Sigmoid5(nn.Module):
#     def forward(self, input):
#         return 5*torch.sigmoid(input)

# MF = MatrixFactorization(paired_train_data, user_list=unique_users, item_list=unique_items, nb_factor=50, lr=1e-2,
#                          weight_decay=0., batch_size=64, drop_rate=0.2, pro_process=Sigmoid5())

MF = MatrixFactorization(paired_train_data, user_list=unique_users, item_list=unique_items, nb_factor=80, lr=1e-3,
                         weight_decay=1e-6, batch_size=64, drop_rate=0.2, pro_process=None)

train_loss, test_loss = MF.train(epochs=80, test_data=list(zip(test_data['userId'].values, test_data['movieId'].values,
                                                               test_data['rating'].values)), test_epoch_step=1)

# top 10 recommendation (id and score).
test_users, test_items = test_data['userId'].values, test_data['movieId'].values
test_unique_users = list(set(test_users))
id_recommendation, rating_recommendation = MF.recommend(test_unique_users, 10)
prediction = MF.predict_ratings(paired_test_data)

a = MF.model.get_rating_matrix().data.cpu().numpy()
b = MF.history_rating_matrix.values

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
    pred_items = set(MF.indices_to_items(id_recommendation[i].ravel()))
    hit = pred_items & test_dict[user]
    precision += len(hit) / len(pred_items)
    recall += len(hit) / len(test_dict[user])

precision /= len(test_unique_users)
recall /= len(test_unique_users)

# using MSE to evaluate the performance
loss = np.mean((prediction-test_ground_truth)**2)
print("The test MSE loss of rating is {}.".format(loss))
print('precision={}, recall={}'.format(precision, recall))
