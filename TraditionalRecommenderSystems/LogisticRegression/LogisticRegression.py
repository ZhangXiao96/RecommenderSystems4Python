from lib.utils import top_k
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR


class BaseLogisticRegression(object):
    def __init__(self, user_item_pairs, user_description, item_description,
                 interactions=None, default_interaction=None, C=1., max_iter=100,
                 class_weight=None):
        """
        Logistic Regression (Directly use LR in sklearn).
        :param user_item_pairs: list. [(user, item, rating)].
        :param C: float. L2 regularization.
        """
        self.user_item_pairs = pd.DataFrame(user_item_pairs)
        self.user_description = user_description
        self.item_description = item_description
        self.interactions = interactions
        self.default_interaction = default_interaction
        if (self.interactions is not None) and (self.default_interaction is None):
            raise Exception('The default_interaction should be set when interactions is not None!')
        self.nb_user, self.nb_item = len(user_description), len(item_description)
        self.model = LR(C=C, max_iter=max_iter, class_weight=class_weight)

        # build index-user, index-item
        self.index_2_user = np.array(list(self.user_description.keys()))
        self.index_2_item = np.array(list(self.item_description.keys()))
        assert len(self.index_2_user) == len(set(self.index_2_user))
        assert len(self.index_2_item) == len(set(self.index_2_item))
        self.user_2_index = {self.index_2_user[i]: i for i in range(len(self.index_2_user))}
        self.item_2_index = {self.index_2_item[i]: i for i in range(len(self.index_2_item))}
        self.nb_user, self.nb_item = len(self.index_2_user), len(self.index_2_item)

    def pair_to_x(self, pair):
        """
        (user, item) to a vector description
        :param pair: (user, item).
        :return: a vector
        """
        if self.interactions:
            if pair[0] in self.interactions and pair[1] in self.interactions[pair[0]]:
                interaction = self.interactions[pair[0]][pair[1]]
            else:
                interaction = self.default_interaction
            return np.concatenate([self.user_description[pair[0]],
                                   self.item_description[pair[1]],
                                   interaction], axis=-1)
        else:
            return np.concatenate([self.user_description[pair[0]],
                                   self.item_description[pair[1]]], axis=-1)

    def pairs_to_xs(self, pairs):
        """
        [(user, item),...] to [x,...]
        :param pairs: list of (user, item)
        :return: ndarray. size=(len(pairs), features)
        """
        return np.array([self.pair_to_x(pair) for pair in pairs])

    def train(self):
        """
        Train the model.
        :return: self.
        """
        X = self.pairs_to_xs(self.user_item_pairs[[0, 1]].values)
        self.model.fit(X, self.user_item_pairs[2].values)
        return self

    def predict(self, user_item_pairs):
        """
        Predict the ratings of the pairs of (user, item).
        :param user_item_pairs: list of (user, item).
        :return: ratings. size=[nb_pairs]
        """
        return self.model.predict(self.pairs_to_xs(user_item_pairs))

    def eval(self, user_item_pairs, ground_truth):
        """
        Eval the accuracy of the pairs of (user, item).
        :param user_item_pairs: list of (user, item).
        :param ground_truth: list of labels.
        :return: accuracy.
        """
        outputs = self.predict(user_item_pairs).ravel()
        acc = np.mean(np.where(outputs == ground_truth, 1., 0.))
        return acc

    def predict_proba(self, user_item_pairs):
        """
        Predict the ratings of the pairs of (user, item).
        :param user_item_pairs: list of (user, item)
        :return: ratings. size=[nb_pairs]
        """
        return self.model.predict_proba(self.pairs_to_xs(user_item_pairs)).ravel()

    # def recommend(self, users, nb_recommendation):
    #     """
    #     return the recommendations and their corresponding ratings.
    #     :param users: list of users
    #     :param nb_recommendation: The number of items to be recommended.
    #     :return: Indices of recommended items and their corresponding scores.
    #     """
    #     all_items = list(self.item_description.keys())
    #     res = []
    #     for user in users:
    #         probs = self.predict_proba([(user, item) for item in all_items])
    #
    #     id_recommend, rating_recommend = top_k(np.where(np.isnan(self.pred_rating_matrix[user_indices, :]),
    #                                                     -np.inf, self.pred_rating_matrix[user_indices, :]),
    #                                            k=nb_recommendation, axis=-1, reverse=True, sort=True)
    #     return id_recommend, rating_recommend

    def users_to_indices(self, users):
        return np.array([self.user_2_index[user] for user in users]).ravel()

    def indices_to_users(self, indices):
        return self.index_2_user[np.array(indices).ravel()]

    def items_to_indices(self, items):
        return np.array([self.item_2_index[item] for item in items]).ravel()

    def indices_to_items(self, indices):
        return self.index_2_item[np.array(indices).ravel()]
