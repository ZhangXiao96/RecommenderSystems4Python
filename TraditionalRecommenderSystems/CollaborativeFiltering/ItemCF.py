from lib.utils import top_k
from lib.metric import cosine
import numpy as np
import pandas as pd
import warnings


class ItemCF(object):
    def __init__(self, user_item_pairs, user_list, item_list, nb_similar_item, fillna='constant', value=0, similarity=cosine):
        """
        Item Collaborative Filtering.
        :param user_item_pairs: list. [(user, item, rating)].
        :param user_list: list. The list of all the users (with no repeat).
        :param item_list: list. The list of all the items (with no repeat).
        :param nb_similar_item: The number of similar items to be utilized.
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
        self.average_scores = None      # This is used to get the scores for ranking instead of rating
        self.filled_history_rating_matrix = None
        self.item_mean = None
        self.user_mean = None
        self.item_sim_matrix = None
        self.pred_rating_matrix = None

        self.update_history_rating_matrix(fillna=fillna, value=value)
        self.update_item_sim_matrix(similarity=similarity)
        self.update_pred_rating_matrix(nb_similar_item=nb_similar_item)

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

    def update_item_sim_matrix(self, similarity=cosine):
        """
        Update item similarity matrix.
        :param similarity: function. Metric.
        :return: self.
        """
        self.item_sim_matrix = similarity(self.filled_history_rating_matrix.values.T,
                                          self.filled_history_rating_matrix.values.T)
        return self

    def update_pred_rating_matrix(self, nb_similar_item):
        """
        Update prediction rating matrix.
        :param nb_similar_item: int. The number of similar items to be utilized.
        :return: self.
        """
        index_top_k, sim_top_k = self.top_k_sim_items(self.index_2_item, nb_similar_item)

        # only keep the similarity of the topK items.
        top_k_masked_sim_matrix = np.zeros(self.item_sim_matrix.shape)
        np.put_along_axis(top_k_masked_sim_matrix, index_top_k, sim_top_k, axis=-1)
        self.average_scores = self.filled_history_rating_matrix @ top_k_masked_sim_matrix
        average_ratings = self.average_scores / (np.where(self.history_rating_matrix.isna(), 0, 1) @
                                                 top_k_masked_sim_matrix + 1e-10)

        self.average_scores = np.where(self.history_rating_matrix.isna(), self.average_scores, np.nan)
        self.pred_rating_matrix = np.where(self.history_rating_matrix.isna(), average_ratings, np.nan)
        return self

    def top_k_sim_items(self, items, nb_similar_item):
        """
        :param items: array. size = [nb_items]
        :param nb_similar_item: int. K.
        :return: index and similarity. size=[nb_items, K]
        """
        item_indices = self.items_to_indices(items)
        index_top_k, sim_top_k = top_k(self.item_sim_matrix[item_indices, :], k=nb_similar_item+1, axis=-1,
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
        id_recommend, rating_recommend = top_k(np.where(np.isnan(self.average_scores[user_indices, :]),
                                                        -np.inf, self.average_scores[user_indices, :]),
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
