from lib.utils import top_k
from lib.metric import cosine
import numpy as np
import pandas as pd
import warnings


class UserCF(object):
    def __init__(self, user_item_matrix, fill_mode='constant', value=0):
        """
        User Collaborative Filtering.
        :param user_item_matrix: Co-occurence matrix. [nb_user, nb_item].
        :param fill_mode: Mode to fill NaN in user_item_matrix. Chosen from {"item_mean", "user_mean", "value"}.
        :param value: only valid when "fill_mode" equals 'constant'.
        """
        self.data = pd.DataFrame(user_item_matrix)
        self.fill_mode = fill_mode
        self.value = value
        if fill_mode == 'item_mean':
            self.item_mean = self.data.mean(axis=0)
            self.filled_data = self.data.fillna(self.item_mean)
        elif fill_mode == 'user_mean':
            self.filled_data = self.data.T.fillna(self.data.mean(axis=-1)).T
        elif fill_mode == 'constant':
            self.filled_data = self.data.fillna(value)
        else:
            raise Exception('The param fill_model should be chosen from {"item_mean", "user_mean", "value"}')
        if self.filled_data.isnull().values.any():
            warnings.warn('NaN still exists. This may happen when using '
                          'item_mean or user_mean but all the values are NaN. '
                          'In this case, \'constant\' fill model is utilized after the first filling.')
            self.filled_data = self.filled_data.fillna(value)

    def recommend(self, test_users, k, nb_recommendation, metric=cosine):
        """
        return the recommendations and their corresponding scores.
        :param test_users: Users to recommend. [nb_test_user, nb_items]
        :param k: The number of similar users to find.
        :param nb_recommendation: The number of items to be recommended.
        :param metric: Similarity of users.
        :return: Indices of recommended items， their corresponding scores and all the scores.
        """
        test_users = pd.DataFrame(test_users)
        if self.fill_mode == 'item_mean':
            filled_test_users = test_users.fillna(self.item_mean)
        elif self.fill_mode == 'user_mean':
            filled_test_users = test_users.T.fillna(self.test_users.mean(axis=-1)).T
        elif self.fill_mode == 'constant':
            filled_test_users = test_users.fillna(self.value)
        if filled_test_users.isnull().values.any():
            warnings.warn('NaN still exists. This may happen when using '
                          'item_mean or user_mean but all the values are NaN. '
                          'In this case, \'constant\' fill model is utilized after the first filling.')
            filled_test_users = filled_test_users.fillna(self.value)

        similarity = metric(filled_test_users.values, self.filled_data.values)  # [test_users, nb_user]
        users_top_k, sim_top_k = top_k(similarity, k=k, axis=-1, reverse=True, sort=False)  # [test_users, k]
        average_scores = np.array([np.sum(sim_top_k[i, :, np.newaxis] *
                                          self.filled_data.values[users_top_k[i], :], axis=0)
                                   / np.sum(sim_top_k[i]) for i in range(len(test_users))])    # [test_users, nb_item]
        filled_average_scores = np.where(test_users.isna(), average_scores, -np.inf)
        id_recommend, score_recommend = top_k(filled_average_scores, k=nb_recommendation, axis=-1, reverse=True, sort=True)
        return id_recommend, score_recommend, average_scores

