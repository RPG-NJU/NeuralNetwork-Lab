# 在时间序列的任务中，需要配置的一些参数
# 包括路径和一些超参数等


import os


class MLPConfig:

    def __init__(self):
        self.PROJECT_PATH = os.getcwd()
        self.TRAIN_DATA_PATH = os.path.join(self.PROJECT_PATH, "Data/train.csv")
        self.TEST_DATA_PATH = os.path.join(self.PROJECT_PATH, "Data/test.csv")

        # 超参数
        self.INPUT_N = 100
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

        # 对数据进行设置
        self.TARGET_SEQ_LEN = 56    # 作为输出目标的序列长度
        self.VALIDATION_SEQ_LEN = self.TARGET_SEQ_LEN   # 不训练的验证集的长度
        self.INPUT_SIZE = 100
        self.TARGET_SIZE = 56    # 每次输出的长度
        self.BATCH_SIZE = 40   # 对于LSTM来说每一个BATCH的大小

        # 对网络进行设置
        self.LAYER_NUM = 1
        self.HIDDEN_DIM = 200
        self.DROPOUT = 0.0

        # 优化器设置
        # 使用SGD
        # self.OPTIMIZER = "SGD"
        # self.LEARN_RATE = 0.05
        # self.MOMENTUM = 0.9
        # self.WEIGHT_DECAY = 0.0
        # 使用Adam
        self.OPTIMIZER = "ADAM"
        self.LEARN_RATE = 0.006
        self.WEIGHT_DECAY = 0.00

        # 损失函数设置
        self.LOSS_FUNCTION = "MSE"
        # self.LOSS_FUNCTION = "L1"

        # 训练设置
        self.EPOCH = 40



