from lib.utils import top_k
from lib.metric import cosine
import numpy as np
import pandas as pd
import warnings


class ItemCF(object):
    def __init__(self, user_item_matrix, fill_mode='constant', value=0, metric=cosine):
        """
        Item Collaborative Filtering.
        :param user_item_matrix: Co-occurence matrix. [nb_user, nb_item].
        :param fill_mode: Mode to fill NaN in user_item_matrix. Chosen from {"item_mean", "user_mean", "value"}.
        :param value: only valid when "fill_mode" equals 'constant'.
        """
        self.data = pd.DataFrame(user_item_matrix)
        self.fill_mode = fill_mode
        self.value = value
        self.metric = metric
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

        # build item-similarity matrix
        self.item_similarity_matrix = metric(self.filled_data.T, self.filled_data.T)  # [nb_item, nb_item]
        self.id_sort = np.argsort(-self.item_similarity_matrix, axis=-1)

    def recommend(self, test_users, k, nb_recommendation):
        """
        return the recommendations and their corresponding scores.
        :param test_users: Users to recommend. [nb_test_user, nb_items]
        :param k: The number of similar items to find.
        :param nb_recommendation: The number of items to be recommended.
        :return: Indices of recommended items， their corresponding scores and all the scores.
        """
        if k >= self.id_sort.shape[-1]:
            raise Exception('k should be small than N_item, which is {} in this case!'.
                            format(self.id_sort.shape[-1]))
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

        # only keep the similarity of the topK items.
        top_k_masked_sim_matrix = self.item_similarity_matrix.copy()
        np.put_along_axis(top_k_masked_sim_matrix, self.id_sort[:, 0:1], 0, axis=-1)  # Note: remove the item itself.
        if k < self.id_sort.shape[-1]-1:
            np.put_along_axis(top_k_masked_sim_matrix, self.id_sort[:, k+1:], 0, axis=-1)  # remove the non-topK items
        average_scores = filled_test_users.values @ top_k_masked_sim_matrix\
                         / (np.where(test_users.isna(), 0, 1) @ top_k_masked_sim_matrix + 1e-10)

        filled_average_scores = np.where(test_users.isna(), average_scores, -np.inf)
        id_recommend, score_recommend = top_k(filled_average_scores, k=nb_recommendation, axis=-1, reverse=True,
                                              sort=True)
        return id_recommend, score_recommend, average_scores
