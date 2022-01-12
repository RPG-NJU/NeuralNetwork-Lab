import pandas as pd
import numpy as np
from config import Config
import torch
from torch.utils.data import Dataset, DataLoader


class OpData:
    @classmethod
    def read_csv(cls, filepath: str):
        """
        阅读一个时间序列的csv文件
        :param filepath: .csv file path
        :return:
        """
        pd_data = pd.read_csv(filepath, header=None, sep=",")
        np_data = pd_data.values
        return np_data.astype(np.float32)


class MyDataset(Dataset):
    def __init__(self, data: np.ndarray):
        super(MyDataset, self).__init__()
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.data.shape[0]


def line_minmax_norm(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    进行行归一化
    :param x: 训练集部分，用于计算min-max并且进行归一化。
    :param y: 使用训练及得到的min-max进行归一化。
    :return: x，y，以及每一行使用的min-max的数值
    """
    min = np.min(x, axis=1).reshape(x.shape[0], 1)
    max = np.max(x, axis=1).reshape(x.shape[0], 1)

    x = (x - min) / (max - min)
    y = (y - min) / (max - min)

    # min_max_list = list()
    # for i in range(0, x.shape[0]):
    #     pair = list()
    #     pair.append(min[i, 0])
    #     pair.append(max[i, 0])
    #     min_max_list.append(pair)

    return x, y, min, max


def line_minmax_denorm(x: np.ndarray, y: np.ndarray, min: np.ndarray, max: np.ndarray) -> (np.ndarray, np.ndarray):
    x = x * (max - min) + min
    y = y * (max - min) + min
    return x, y


def get_windows_data(x: np.ndarray, window_size: int) -> np.ndarray:
    """
    将输入的数据，按照滑动窗口的方式得到滑动之后拼接的数据集
    :param x:
    :param window_size:
    :return: 所有滑动窗口拼接而成的数据集
    """
    # print(x.shape)
    result = np.zeros((0, window_size))
    for i in range(0, x.shape[1]-window_size):
        result = np.concatenate((result, x[:, i: i+window_size]), axis=0)
    return result.astype(np.float32)


if __name__ == '__main__':
    c = Config()
    train_data = OpData.read_csv(c.TRAIN_DATA_PATH)
    dataset = MyDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    # for a_batch_data in dataloader:
    #     print(a_batch_data)
    # a_batch_data = dataiter.__next__()
    # print(a_batch_data.shape)
    print(train_data)
    norm_data, _, min, max = line_minmax_norm(train_data, train_data)
    print(norm_data)
    denorm_data, _ = line_minmax_denorm(norm_data, norm_data, min, max)
    print(denorm_data)
    windows_data = get_windows_data(train_data, 50)
    loader = DataLoader(windows_data, batch_size=10, shuffle=True)
    # for data in loader:
    #     print(data)