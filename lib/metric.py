import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine(users1, users2):
    return cosine_similarity(users1, users2)


def pcc_without_user_bias(users1, users2):
    users1 = users1 - np.mean(users1, axis=-1, keepdims=True)
    users2 = users2 - np.mean(users2, axis=-1, keepdims=True)
    return cosine_similarity(users1, users2)


def pcc_without_item_bias(users1, users2):
    users1 = users1 - np.mean(users1, axis=0, keepdims=True)
    users2 = users2 - np.mean(users2, axis=0, keepdims=True)
    return cosine_similarity(users1, users2)
