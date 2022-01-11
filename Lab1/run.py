# 运行文件即可得到实验结果

from opData import OpData
from config import Config
from MLP import BaselineMLP


def run_baseline():
    baseline_config = Config()
    baseline_mlp = BaselineMLP(baseline_config)
    for epoch in range(0, baseline_config.EPOCH):
        loss, acc = baseline_mlp.train()
        print(loss, acc)


if __name__ == '__main__':
    print("===>  RUN")
    run_baseline()