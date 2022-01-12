# 用于训练LSTM的训练器类别
# 及相关函数

from opData import *
from config import LSTMConfig


class Trainer:
    def __init__(self, config=LSTMConfig()):
        # 对Train和Test数据进行获取
        self.train_data = OpData.read_csv(config.TRAIN_DATA_PATH)
        self.test_data = OpData.read_csv(config.TEST_DATA_PATH)
        # 同时得到一些归一化的数据，另外保存如下，以及相对应的min-max数据
        self.train_data_norm, self.test_data_norm, self.min, self.max = line_minmax_norm(self.train_data, self.test_data)
