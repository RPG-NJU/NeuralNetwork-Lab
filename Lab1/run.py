# 运行文件即可得到实验结果

from opData import OpData
from config import Config
from MLP import BaselineMLP
import tensorboardX as tb


def run():
    config = Config()
    mlp = BaselineMLP(config)
    tb_writer = tb.SummaryWriter(config.TB_LOG_PATH)
    for epoch in range(0, config.EPOCH):
        print("===>  Epoch: %d" % epoch)
        loss, acc = mlp.train()
        print("Train Set:, Loss=%f, Acc=%f" % (loss, acc))
        test_acc = mlp.test()
        print("Test Set: Acc=%f" % test_acc)
        tb_writer.add_scalar("Loss", loss, epoch)
        tb_writer.add_scalars("Acc",
                              {"Train": acc, "Test": test_acc},
                              epoch)


if __name__ == '__main__':
    print("===>  RUN")
    run()
