# 增加一些通用的模型细节
# 作为父类进行继承，而不直接使用

import torch.nn as nn
from config import MLPConfig
from opData import *
from torch.utils.data import Dataset, DataLoader


class MyModule(nn.Module):
    def __init__(self, config: MLPConfig()):
        super(MyModule, self).__init__()

        self.batch_size = config.BATCH_SIZE

        self.train_data = OpData.read_csv(config.TRAIN_DATA_PATH)
        self.test_data = OpData.read_csv(config.TEST_DATA_PATH)

        self.input_num = config.INPUT_N
        self.output_num = config.OUTPUT_N
        self.learn_rate = config.LEARN_RATE
        self.momentum = config.MOMENTUM

        self.train_index = 0
        self.validation_index = self.train_data.shape[1] - self.output_num
        self.window_index = 0  # 用于标记滑动窗口的位置

        train_data_norm, self.min, self.max = OpData.line_norm(self.train_data[:, :self.validation_index])
        validation_data_norm, _, _ = OpData.line_norm(self.train_data[:, self.validation_index:], line_min=self.min, line_max=self.max)

        self.train_data = np.concatenate((train_data_norm, validation_data_norm), axis=1)
        self.test_data, _, _ = OpData.line_norm(self.test_data, line_min=self.min, line_max=self.max)
        # self.train_data, self.test_data, self.min, self.max = line_minmax_norm(self.train_data, self.test_data)


