import numpy as np


def top_k(matrix, k=5, axis=-1, reverse=False, sort=False):
    """
    Find the topK indices and values.
    :param matrix: array size=[N1, N2, N3, ...]. Input matrix to find topK.
    :param k: int. K.
    :param axis: int. Along which axis to perform topK.
    :param reverse: boolean. True (False) for finding the k largest (least) values.
    :param sort: boolean. Whether the output K elements are sorted.
    :return: index_top_k, value_top_k.
    """
    if k > matrix.shape[axis]:
        raise Exception('{} is larger than the size of the matrix({})!'.format(k, matrix.shape[axis]))
    if reverse:
        index_top_k = np.argpartition(matrix, kth=-k, axis=axis)
        index_top_k = np.take(index_top_k, np.arange(matrix.shape[axis]-k, matrix.shape[axis]), axis=axis)
    else:
        index_top_k = np.argpartition(matrix, kth=k, axis=axis)
        index_top_k = np.take(index_top_k, np.arange(k), axis=axis)
    value_top_k = np.take_along_axis(matrix, index_top_k, axis=axis)
    if not sort:
        return index_top_k, value_top_k
    else:
        if reverse:
            sorted_index = np.take_along_axis(index_top_k, np.argsort(-value_top_k, axis=axis), axis=axis)
            return sorted_index, np.take_along_axis(matrix, sorted_index, axis=axis)
        else:
            sorted_index = np.take_along_axis(index_top_k, np.argsort(value_top_k, axis=axis), axis=axis)
            return sorted_index, np.take_along_axis(matrix, sorted_index, axis=axis)


if __name__ == "__main__":
    a = np.array([[[4, 2, 3, 5, 1, 9, 8, 3.2], [9, 8, 3.6, 4, 2, 3, 5, 1]], [[4, 2, 3, 5, 1, 9, 8, 3.2], [9, 8, 3.6, 4, 2, 3, 5, 1]]])
    print(top_k(a, 5, axis=-1, reverse=True, sort=True))
