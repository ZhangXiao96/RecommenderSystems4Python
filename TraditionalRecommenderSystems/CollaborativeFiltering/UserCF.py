from lib.utils import top_k
from lib.metric import cosine
import numpy as np
import pandas as pd
import warnings


class UserCF(object):
    def __init__(self, user_item_pairs, user_list, item_list, nb_similar_user, fillna='constant', value=0, similarity=cosine):
        """
        User Collaborative Filtering.
        :param user_item_pairs: [(user, item, rating)].
        :param user_list: list. The list of existing users.
        :param user_list: list. The list of existing items.
        :param nb_similar_user: The number of similar users to be utilized.
        :param fillna: mode to fill NaN in user_item_matrix. Chosen from {"item_mean", "user_mean", "constant"}.
        :param value: only valid when "fillna" equals 'constant'.
        :param similarity: similarity metric.
        """
        self.user_item_pairs = pd.DataFrame(user_item_pairs)

        # build index-user, index-item
        self.index_2_user = np.array(user_list)
        self.index_2_item = np.array(item_list)
        assert len(self.index_2_user) == len(set(self.index_2_user))
        assert len(self.index_2_item) == len(set(self.index_2_item))
        self.user_2_index = {self.index_2_user[i]: i for i in range(len(self.index_2_user))}
        self.item_2_index = {self.index_2_item[i]: i for i in range(len(self.index_2_item))}
        self.nb_user, self.nb_item = len(user_list), len(item_list)

        self.history_rating_matrix = None
        self.filled_history_rating_matrix = None
        self.item_mean = None
        self.user_mean = None
        self.user_sim_matrix = None
        self.pred_rating_matrix = None

        self.update_history_rating_matrix(fillna=fillna, value=value)
        self.update_user_sim_matrix(similarity=similarity)
        self.update_pred_rating_matrix(nb_similar_user=nb_similar_user)

    def update_history_rating_matrix(self, fillna='constant', value=0):
        """
        Update history rating matrix.
        :param fillna: mode to fill NaN in user_item_matrix. Chosen from {"item_mean", "user_mean", "constant"}.
        :param value: only valid when "fillna" equals 'constant'.
        :return: self.
        """
        self.history_rating_matrix = pd.DataFrame(index=self.index_2_user, columns=self.index_2_item)
        for i, j, k in self.user_item_pairs.values:
            if i and j and k:
                self.history_rating_matrix[j][i] = k

        if fillna == 'item_mean':
            self.filled_history_rating_matrix = self.history_rating_matrix.fillna(self.item_mean)
        elif fillna == 'user_mean':
            self.filled_history_rating_matrix = self.history_rating_matrix.T.fillna(self.user_mean).T
        elif fillna == 'constant':
            self.filled_history_rating_matrix = self.history_rating_matrix.fillna(value)
        else:
            raise Exception('The param fill_model should be chosen from {"item_mean", "user_mean", "value"}')
        if self.filled_history_rating_matrix.isnull().values.any():
            warnings.warn('NaN still exists. This may happen when using '
                          'item_mean or user_mean but all the values are NaN. '
                          'In this case, \'constant\' fill model is utilized after the first filling.')
            self.filled_history_rating_matrix = self.filled_history_rating_matrix.fillna(value)
        return self

    def update_user_sim_matrix(self, similarity=cosine):
        """
        Update user similarity matrix.
        :param similarity: function. Metric.
        :return: self.
        """
        self.user_sim_matrix = similarity(self.filled_history_rating_matrix.values,
                                          self.filled_history_rating_matrix.values)
        return self

    def update_pred_rating_matrix(self, nb_similar_user):
        """
        Update prediction rating matrix.
        :param nb_similar_user: int. The number of similar users to be utilized.
        :return: self.
        """
        index_top_k, sim_top_k = self.top_k_sim_users(self.index_2_user, nb_similar_user)
        average_ratings = np.array([np.sum(sim_top_k[i, :, np.newaxis] *
                                          self.filled_history_rating_matrix.values[index_top_k[i], :], axis=0)
                                   / np.sum(sim_top_k[i]) for i in range(len(self.index_2_user))])
        self.pred_rating_matrix = np.where(self.history_rating_matrix.isna(), average_ratings, np.nan)
        return self

    def top_k_sim_users(self, users, nb_similar_user):
        """
        :param users: array. size = [nb_users]
        :param nb_similar_user: int. K.
        :return: index and similarity. size=[nb_users, K]
        """
        user_indices = self.users_to_indices(users)
        index_top_k, sim_top_k = top_k(self.user_sim_matrix[user_indices, :], k=nb_similar_user+1, axis=-1,
                                       reverse=True, sort=True)  # [test_users, k]
        return index_top_k[:, 1:], sim_top_k[:, 1:]

    def get_single_rating(self, i, j):
        return self.pred_rating_matrix[i][j] if not np.isnan(self.pred_rating_matrix[i][j])\
            else self.history_rating_matrix.values[i][j]

    def predict_ratings(self, user_item_pairs):
        """
        Predict the ratings of the pairs of (user, item).
        :param user_item_pairs: list of (user, item)
        :return: ratings. size=[nb_pairs]
        """
        pairs = pd.DataFrame(user_item_pairs)
        users = self.users_to_indices(pairs[0])
        items = self.items_to_indices(pairs[1])
        return np.array([self.get_single_rating(users[i], items[i]) for i in range(len(user_item_pairs))])

    def recommend(self, users, nb_recommendation):
        """
        return the recommendations and their corresponding ratings.
        :param users: array of users
        :param nb_recommendation: The number of items to be recommended.
        :return: Indices of recommended items and their corresponding scores.
        """
        user_indices = self.users_to_indices(users)
        id_recommend, rating_recommend = top_k(np.where(np.isnan(self.pred_rating_matrix[user_indices, :]),
                                                        -np.inf, self.pred_rating_matrix[user_indices, :]),
                                               k=nb_recommendation, axis=-1, reverse=True, sort=True)
        return id_recommend, rating_recommend

    def users_to_indices(self, users):
        return np.array([self.user_2_index[user] for user in users]).ravel()

    def indices_to_users(self, indices):
        return self.index_2_user[np.array(indices).ravel()]

    def items_to_indices(self, items):
        return np.array([self.item_2_index[item] for item in items]).ravel()

    def indices_to_items(self, indices):
        return self.index_2_item[np.array(indices).ravel()]
