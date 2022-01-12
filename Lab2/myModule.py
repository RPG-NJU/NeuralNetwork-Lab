# 增加一些通用的模型细节
# 作为父类进行继承，而不直接使用

import torch.nn as nn
from config import Config
from opData import *
from torch.utils.data import Dataset, DataLoader


class MyModule(nn.Module):
    def __init__(self, config: Config()):
        super(MyModule, self).__init__()

        self.batch_size = config.BATCH_SIZE

        self.train_data = OpData.read_csv(config.TRAIN_DATA_PATH)
        self.test_data = OpData.read_csv(config.TEST_DATA_PATH)

        self.train_data, self.test_data, self.min, self.max = line_minmax_norm(self.train_data, self.test_data)


