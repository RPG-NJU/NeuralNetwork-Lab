import pandas as pd
import numpy as np
from config import MLPConfig
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

    @classmethod
    def line_norm(cls, x: np.ndarray, line_min=None, line_max=None):
        if line_min is None:
            line_min = np.min(x, axis=1).reshape(x.shape[0], 1)
            line_max = np.max(x, axis=1).reshape(x.shape[0], 1)
        x = (x - line_min) / (line_max - line_min)
        return x, line_min, line_max

    @classmethod
    def get_norm_minmax(cls, x: np.ndarray):
        """
        根据这一段数据，生成min max的列表
        :param x: 归一化的依据
        :return:
        """
        line_min = np.min(x, axis=1).reshape(x.shape[0], 1)
        line_max = np.max(x, axis=1).reshape(x.shape[0], 1)
        return line_min, line_max

    @classmethod
    def line_denorm(cls, x: np.ndarray, line_min, line_max):
        return x * (line_max - line_min) + line_min


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


class SeqWindowDataset(Dataset):
    def __init__(self, seq_data: np.ndarray, input_size: int, target_size: int, target_seq_len: int):
        """
        初始化一个滑动窗口的Dataset类别
        :param seq_data:
        :param input_size:
        :param target_size:
        :param target_seq_len:
        """
        self.seq_data = seq_data
        self.input_size = input_size
        self.target_size = target_size
        self.target_seq_len = target_seq_len

    def __len__(self):
        return self.seq_data.shape[1] - self.target_seq_len

    def __getitem__(self, item) -> (np.ndarray, np.ndarray):
        """
        返回一个索引开始为item的滑动窗口
        :param item:
        :return:
        """
        input_seq = self.seq_data[:, item: item + self.input_size]
        target_seq = self.seq_data[:, item + self.input_size: item + self.target_size]
        return input_seq, target_seq


if __name__ == '__main__':
    c = MLPConfig()
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