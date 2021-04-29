"""
This implementation references to https://github.com/EthanRosenthal/torchmf/blob/master/torchmf.py.
"""
import torch
from torch import nn


class BaseMF(nn.Module):
    def __init__(self, nb_user, nb_item, nb_factor, drop_rate=0.5):
        super(BaseMF, self).__init__()
        self.nb_user, self.nb_item, self.nb_factor = nb_user, nb_item, nb_factor
        self.user_biases, self.item_biases = nn.Embedding(nb_user, 1), nn.Embedding(nb_item, 1)
        self.global_bias = torch.Tensor(1)
        self.user_embeddings = nn.Embedding(nb_user, nb_factor)
        self.item_embeddings = nn.Embedding(nb_item, nb_factor)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, users, items):
        user_matrix = self.user_embeddings(users)
        item_matrix = self.item_embeddings(items)
        preds = (self.dropout(user_matrix) * self.dropout(item_matrix)).sum(axis=-1, keepdim=True)
        preds += self.user_biases(users) + self.item_biases(items) + self.global_bias
        return preds

    def predict(self, users, items):
        return self.forward(users, items)

    def get_rating_matrix(self):
        return self.user_embeddings.weight @ self.item_embeddings.weight.t()
