# 项目中需要的配置文件
# 例如超参数的设置，或是文件目录等信息
# 可以直接在Config类中进行调整

import os
import sys


class Config:

    def __init__(self):
        self.project_path = os.getcwd()     # 获取当前运行目录
        self.train_images_path = os.path.join(self.project_path, "Data/train-images-idx3-ubyte")
        self.train_labels_path = os.path.join(self.project_path, "Data/train-labels-idx1-ubyte")
        self.test_images_path = os.path.join(self.project_path, "Data/t10k-images-idx3-ubyte")
        self.test_labels_path = os.path.join(self.project_path, "Data/t10k-labels-idx1-ubyte")


if __name__ == '__main__':
    CONFIG = Config()
    print(CONFIG.project_path)
    print(CONFIG.train_images_path)

