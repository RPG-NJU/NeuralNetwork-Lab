# 项目中需要的配置文件
# 例如超参数的设置，或是文件目录等信息
# 可以直接在Config类中进行调整

import os
import sys


class Config:

    def __init__(self):
        self.PROJECT_PATH = os.getcwd()     # 获取当前运行目录
        self.TRAIN_IMAGES_PATH = os.path.join(self.PROJECT_PATH, "Data/train-images-idx3-ubyte")
        self.TRAIN_LABELS_PATH = os.path.join(self.PROJECT_PATH, "Data/train-labels-idx1-ubyte")
        self.TEST_IMAGES_PATH = os.path.join(self.PROJECT_PATH, "Data/t10k-images-idx3-ubyte")
        self.TEST_LABELS_PATH = os.path.join(self.PROJECT_PATH, "Data/t10k-labels-idx1-ubyte")
        self.TB_LOG_PATH = os.path.join(self.PROJECT_PATH, "tb")

        self.INIT_LEARN_RATE = 0.02
        self.BATCH_SIZE = 20
        self.EPOCH = 30

        self.REG_MODE = "None"
        self.WEIGHT_DECAY = 0.0


if __name__ == '__main__':
    CONFIG = Config()
    # print(CONFIG.project_path)
    # print(CONFIG.train_images_path)

