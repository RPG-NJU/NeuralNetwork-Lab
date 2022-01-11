import numpy as np


def y_to_labels(y: np.ndarray):
    return np.argmax(y, axis=0)


def labels_equal_num(y: np.ndarray, y_gt: np.ndarray):
    """
    返回标签一致的个数
    :param y: Model output y.
    :param y_gt: y's Ground Truth.
    :return: Equal Num.
    """
    assert y.shape == y_gt.shape
    return np.sum(y == y_gt)


if __name__ == '__main__':
    y_test = [[2, 3], [2, 5], [9, 0]]
    y_test = np.array(y_test)
    print(y_to_labels(y_test).shape)
    print(np.sum(y_test, axis=(0, 1)))
    print(y_test[1:2])

    a = [2, 3, 4, 5]
    b = [2, 1]

    aa = [a, a]
    bb = [b, b]

    a_n = np.array(a).T
    b_n = np.array(b).T
    aa_n = np.array(aa).T
    bb_n = np.array(bb).T

    print(a_n.shape, b_n.shape)

    print(b_n.reshape(2, 1) @ a_n.reshape(4, 1).T)
    print(bb_n @ aa_n.T)
