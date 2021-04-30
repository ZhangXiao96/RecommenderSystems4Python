from lib.utils import top_k
from TraditionalRecommenderSystems.MatrixFactorization.Models import BaseMF
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data
from tqdm import tqdm


class MatrixFactorization(object):
    def __init__(self, user_item_pairs, user_list, item_list, nb_factor=40, drop_rate=0.5, batch_size=32, lr=1e-1,
                 optimizer=torch.optim.Adam, loss_func=nn.MSELoss(reduction='mean'), sparse=False,
                 weight_decay=0., device='cuda'):
        """
        Matrix Factorization.
        :param user_item_pairs: [(user, item, rating)].
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

        # prepare training loader
        train_user_indices = torch.from_numpy(self.users_to_indices(self.user_item_pairs[0].values)).long()
        train_item_indices = torch.from_numpy(self.items_to_indices(self.user_item_pairs[1].values)).long()
        train_ratings = torch.from_numpy(self.user_item_pairs[2].values).float()
        self.train_data_loader = data.DataLoader(data.TensorDataset(train_user_indices, train_item_indices,
                                                                    train_ratings), batch_size=batch_size, shuffle=True)

        # build model
        self.nb_factor = nb_factor
        self.lr = lr
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.weight_decay = weight_decay
        self.device = device
        self.sparse = sparse
        self.model = BaseMF(self.nb_user, self.nb_item, nb_factor, drop_rate, sparse).to(device)
        self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # build history rating matrix
        self.pred_rating_matrix = None
        self.history_rating_matrix = None
        self.update_history_rating_matrix()

    def train(self, epochs, test_data=None, test_epoch_step=1):
        hist_train_loss, hist_test_loss = [], []
        if test_data is not None:
            test_data = pd.DataFrame(test_data)
        for epoch in range(epochs):
            print('Epoch-{}/{}:'.format(epoch+1, epochs))
            self.model.train()
            train_loss = self.train_epoch()
            hist_train_loss.append(train_loss)
            if (test_data is not None) and (epoch % test_epoch_step == 0):
                self.model.eval()
                test_loss = self.eval(test_data.iloc[:, [0, 1]].values, ground_truth=test_data[2].values)
                hist_test_loss.append(test_loss)
                print('training loss = {}, test loss = {}'.format(train_loss, test_loss))
            else:
                print('training loss = {}'.format(train_loss))
        self.update_pred_rating_matrix()
        return hist_train_loss, hist_test_loss

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.
        for id_user, id_item, id_rating in tqdm(self.train_data_loader):
            batch_loss = self.train_on_batch(id_user, id_item, id_rating)
            epoch_loss += batch_loss
        epoch_loss /= len(self.train_data_loader)
        return epoch_loss

    def train_on_batch(self, user_indices, item_indices, ratings):
        users, items, ratings = user_indices.to(self.device), item_indices.to(self.device), ratings.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(users, items)
        loss = self.loss_func(outputs, ratings)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval(self, user_item_pairs, ground_truth, batch_size=100):
        """
        Predict the ratings of the pairs of (user, item).
        :param user_item_pairs: list of (user, item).
        :param ground_truth: the ground truth rating.
        :param batch_size: batch_size of predicting.
        :return: ratings. size=[nb_pairs]
        """
        self.model.eval()
        outputs = self.predict(user_item_pairs, batch_size=batch_size).ravel()
        loss = np.mean((outputs-ground_truth.ravel())**2)
        return loss

    def predict(self, user_item_pairs, batch_size=100):
        """
        Predict the ratings of the pairs of (user, item).
        :param user_item_pairs: list of (user, item)
        :param batch_size: batch_size of predicting.
        :return: ratings. size=[nb_pairs]
        """
        pairs = pd.DataFrame(user_item_pairs)
        user_indices = self.users_to_indices(pairs[0].values)
        item_indices = self.items_to_indices(pairs[1].values)
        self.model.eval()
        outputs = []
        with torch.no_grad():
            start_id = 0
            end_id = min(batch_size, len(pairs))
            while start_id < len(pairs):
                outputs.append(self.predict_on_batch(user_indices[start_id:end_id], item_indices[start_id:end_id]))
                start_id += batch_size
                end_id = min(start_id+batch_size, len(pairs))
        return np.concatenate(outputs, axis=0)

    def predict_on_batch(self, user_indices, item_indices):
        users = torch.from_numpy(user_indices).long().to(self.device)
        items = torch.from_numpy(item_indices).long().to(self.device)
        outputs = self.model(users, items)
        return outputs.data.cpu().numpy()

    def update_history_rating_matrix(self):
        """
        Update history rating matrix.
        :return: self.
        """
        self.history_rating_matrix = pd.DataFrame(index=self.index_2_user, columns=self.index_2_item)
        for i, j, k in self.user_item_pairs.values:
            if i and j and k:
                self.history_rating_matrix[j][i] = k
        return self

    def update_pred_rating_matrix(self):
        """
        Update prediction rating matrix.
        :return: self.
        """
        pred_matrix = self.model.get_rating_matrix().data.cpu().numpy()
        self.pred_rating_matrix = np.where(self.history_rating_matrix.isna(), pred_matrix, np.nan)
        return self

    # def get_single_rating(self, i, j):
    #     return self.pred_rating_matrix[i][j] if not np.isnan(self.pred_rating_matrix[i][j])\
    #         else self.history_rating_matrix.values[i][j]
    #
    # def predict_ratings_with_matrix(self, user_item_pairs):
    #     """
    #     Predict the ratings of the pairs of (user, item).
    #     :param user_item_pairs: list of (user, item)
    #     :return: ratings. size=[nb_pairs]
    #     """
    #     pairs = pd.DataFrame(user_item_pairs)
    #     users = self.users_to_indices(pairs[0])
    #     items = self.items_to_indices(pairs[1])
    #     return np.array([self.get_single_rating(users[i], items[i]) for i in range(len(user_item_pairs))])

    def predict_ratings(self, user_item_pairs):
        """
        Predict the ratings of the pairs of (user, item).
        :param user_item_pairs: list of (user, item)
        :return: ratings. size=[nb_pairs]
        """
        return self.predict(user_item_pairs).ravel()

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
