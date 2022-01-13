# 用于训练LSTM的训练器类别
# 及相关函数

from opData import *
from config import LSTMConfig
import numpy as np


class Trainer:
    def __init__(self, config=LSTMConfig()):
        # 对Train和Test数据进行获取
        self.config = config

        # 从Data中分别读取Train和Test数据
        self.train_data = OpData.read_csv(config.TRAIN_DATA_PATH)
        self.test_data = OpData.read_csv(config.TEST_DATA_PATH)

        # 进行序列划分，得到Train，Validate和Test序列
        self.train_seq_data = self.train_data[:, :self.train_data.shape[1] - self.config.TARGET_SEQ_LEN]
        self.validate_seq_data = self.train_data[:, self.train_data.shape[1] - self.config.TARGET_SEQ_LEN - self.config.INPUT_SIZE:]
        self.test_seq_data = np.concatenate((self.train_data[:, -self.config.INPUT_SIZE:], self.test_data), axis=1)

        # 按照Train Seq来得到归一化使用的min和max
        self.line_min, self.line_max = OpData.get_norm_minmax(self.train_seq_data)

        # 得到归一化的数据
        self.train_seq_norm, _, _ = OpData.line_norm(self.train_seq_data, self.line_min, self.line_max)
        self.validate_seq_norm, _, _ = OpData.line_norm(self.validate_seq_data, self.line_min, self.line_max)
        self.test_seq_norm, _, _ = OpData.line_norm(self.test_seq_data, self.line_min, self.line_max)


if __name__ == '__main__':
    trainer = Trainer()
    print(trainer.train_seq_data.shape, trainer.validate_seq_data.shape, trainer.test_seq_data.shape)

