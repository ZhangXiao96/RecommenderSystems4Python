import numpy as np
import pandas as pd
from TraditionalRecommenderSystems.CollaborativeFiltering.UserCF import UserCF
from TraditionalRecommenderSystems.CollaborativeFiltering.ItemCF import ItemCF
from lib.metric import cosine
import os

# load dataset
data_root = os.path.join('..', 'Dataset', 'MovieLens', 'ml-latest-small')
rating_data_path = os.path.join(data_root, 'ratings.csv')
movie_data_path = os.path.join(data_root, 'movies.csv')

model = 'ItemCF'    # chosen from 'ItemCF' or 'UserCF'

rating_data = pd.read_csv(rating_data_path)
movie_data = pd.read_csv(movie_data_path)
data = pd.merge(rating_data, movie_data, on='movieId')

# build co-occurrence matrix
co_matrix = pd.pivot_table(data, values='rating', index='userId', columns='title').values

train_co_matrix = co_matrix[:500, :]
true_co_matrix = co_matrix[500:, :]

# here we delete the ratings of one movie in test_co_matrix so that we can evaluate the performance of the model.
most_rated_movie = np.argmax(np.sum(np.where(pd.DataFrame(true_co_matrix).isna(), 0, 1), axis=0))    # find the movie rated by most users

# build the input co-matrix
test_co_matrix = true_co_matrix.copy()
test_co_matrix[:, most_rated_movie] = np.nan    # set the ratings of this movie in test co-matrix to NaN

if model == 'UserCF':
    # using UserCF to see the predicted ratings of the movie
    CF = UserCF(train_co_matrix, fill_mode="constant", value=0)
    # top 20 recommendation (id and score) and the whole rating matrix.
    id_recommendation, score_recommendation, ratings = CF.recommend(test_co_matrix, 20, 20, metric=cosine)
elif model == 'ItemCF':
    # using ItemCF to see the predicted ratings of the movie.
    CF = ItemCF(train_co_matrix, fill_mode="constant", value=0, metric=cosine)
    # get top 20 recommendation (id and score) and the whole predicted rating matrix.
    id_recommendation, score_recommendation, ratings = CF.recommend(test_co_matrix, 100, 20)
else:
    raise Exception('No model named {}!'.format(model))

# using MSE to evaluate the performance
error = ratings[:, most_rated_movie] - true_co_matrix[:, most_rated_movie]
error = np.nanmean(error**2)
print("The test MSE loss of the movie \"{}\" is {}.".format(movie_data['title'].values[most_rated_movie], error))
