# 在时间序列的任务中，需要配置的一些参数
# 包括路径和一些超参数等


import os


class MLPConfig:

    def __init__(self):
        self.PROJECT_PATH = os.getcwd()
        self.TRAIN_DATA_PATH = os.path.join(self.PROJECT_PATH, "Data/train.csv")
        self.TEST_DATA_PATH = os.path.join(self.PROJECT_PATH, "Data/test.csv")

        # 超参数
        self.INPUT_N = 200
        self.OUTPUT_N = 56
        self.BATCH_SIZE = 50

        self.LEARN_RATE = 0.01
        self.MOMENTUM = 0.9
        self.EPOCH = 100


class LSTMConfig:
    def __init__(self):
        self.PROJECT_PATH = os.getcwd()
        self.TRAIN_DATA_PATH = os.path.join(self.PROJECT_PATH, "Data/train.csv")
        self.TEST_DATA_PATH = os.path.join(self.PROJECT_PATH, "Data/test.csv")



