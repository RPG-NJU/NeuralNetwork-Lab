# 运行文件即可得到实验结果

from opData import OpData
from config import Config
from MLP import BaselineMLP, InitMethod
import tensorboardX as tb
from layers import *


def run_baseline():
    config = Config()
    config.WEIGHT_DECAY = 0.001
    config.REG_MODE = "L1"
    mlp = BaselineMLP(config)
    # mlp.layer_list.append(Sigmoid())
    # mlp.loss_function = MSELoss()
    tb_writer = tb.SummaryWriter(config.TB_LOG_PATH)
    for epoch in range(0, config.EPOCH):
        print("===>  Epoch: %d" % (epoch+1))
        loss, acc = mlp.train()
        print("Train Set:, Loss=%f, Acc=%f" % (loss, acc))
        test_acc = mlp.test()
        print("Test Set: Acc=%f" % test_acc)
        tb_writer.add_scalar("Loss", loss, epoch)
        tb_writer.add_scalars("Acc",
                              {"Train": acc, "Test": test_acc},
                              epoch)
        # mlp.set_learn_rate(0)


def get_baseline_mlp():
    """
    建立用于对比的Baseline MLP，参数均匀采样随机初始化+交叉熵+固定学习率+不正则化
    :return:
    """
    config = Config()
    return BaselineMLP(config=config)


def get_zero_init_mlp():
    """
    返回一个使用ZERO初始化方法的MLP
    :return:
    """
    config = Config()
    zero_init_mlp = BaselineMLP(config=config)
    zero_init_mlp.init(InitMethod.ZERO)
    return zero_init_mlp


def get_he_init_mlp():
    """
    返回一个使用HE初始化方法的MLP
    :return:
    """
    config = Config()
    he_init_mlp = BaselineMLP(config=config)
    he_init_mlp.init(InitMethod.HE)
    return he_init_mlp


def get_xavier_init_mlp():
    """
    返回一个使用Xivaer初始化方法的MLP
    :return:
    """
    config = Config()
    xavier_init_mlp = BaselineMLP(config=config)
    xavier_init_mlp.init(InitMethod.XAVIER)
    return xavier_init_mlp


def get_mse_mlp():
    """
    返回一个使用了MSE损失函数的MLP
    :return:
    """
    config = Config()
    mse_mlp = BaselineMLP(config=config)
    mse_mlp.layer_list.append(Sigmoid())
    mse_mlp.loss_function = MSELoss()
    return mse_mlp


def get_lr_decay_mlp():
    """
    返回一个学习率衰减的模型，其实本质上是一个Baseline模型，之后依靠外部进行LR调整。
    :return:
    """
    config = Config()
    lr_decay_mlp = BaselineMLP(config=config)
    return lr_decay_mlp


def get_l1_regular_mlp():
    """
    返回一个L1正则化的MLP
    :return:
    """
    config = Config()
    config.WEIGHT_DECAY = 0.001
    config.REG_MODE = "L1"
    return BaselineMLP(config=config)


def get_l2_regular_mlp():
    """
    返回一个L2正则化的MLP
    :return:
    """
    config = Config()
    config.WEIGHT_DECAY = 0.001
    config.REG_MODE = "L2"
    return BaselineMLP(config=config)


def compare_all_method():
    """
    直接运行这一个函数，可以对比多种不同的方法，并且将结果输出在终端和Tensorboard中，用于作图。
    :return: None
    """
    default_config = Config()   # 获得默认的Config信息
    tb_writer = tb.SummaryWriter(default_config.TB_LOG_PATH)

    # 获得所有的MLP模型！并且进行对比实验！
    # Baseline模型
    baseline_mlp = get_baseline_mlp()
    # 各种初始化的方法对比
    zero_init_mlp = get_zero_init_mlp()
    xavier_init_mlp = get_xavier_init_mlp()
    he_init_mlp = get_he_init_mlp()
    # 替换Loss函数为MSE的对比
    mse_mlp = get_mse_mlp()
    # 学习率衰减的模型
    lr_decay_mlp = get_lr_decay_mlp()
    # 对比不同的正则化方法
    l1_reg_mlp = get_l1_regular_mlp()
    l2_reg_mlp = get_l2_regular_mlp()

    # 进行训练和测试并且输出结果，绘制Tensorboard
    for epoch in range(0, default_config.EPOCH):
        # 训练
        if epoch != 0 and epoch % 10 == 0:
            # 每10个Epoch进行一次学习率衰减
            lr_decay_mlp.set_learn_rate(lr_decay_mlp.learn_rate * 0.5)
        baseline_loss, baseline_train_acc = baseline_mlp.train()
        zero_loss, zero_train_acc = zero_init_mlp.train()
        xavier_loss, xavier_train_acc = xavier_init_mlp.train()
        he_loss, he_train_acc = he_init_mlp.train()
        mse_loss, mse_train_acc = mse_mlp.train()
        lrdecay_loss, lrdecay_train_acc = lr_decay_mlp.train()
        l1_loss, l1_train_acc = l1_reg_mlp.train()
        l2_loss, l2_train_acc = l2_reg_mlp.train()

        # 测试
        baseline_test_acc = baseline_mlp.test()
        zero_test_acc = zero_init_mlp.test()
        xavier_test_acc = xavier_init_mlp.test()
        he_test_acc = he_init_mlp.test()
        mse_test_acc = mse_mlp.test()
        lrdecay_test_acc = lr_decay_mlp.test()
        l1_test_acc = l1_reg_mlp.test()
        l2_test_acc = l2_reg_mlp.test()

        print("====>  Current Epoch: %d" % (epoch+1))
        # 输出训练内容
        print("Loss: Baseline=%.6f, ZeroInit=%.6f, XavierInit=%.6f, "
              "HeInit=%.6f, MSELoss=%.6f, LrDecay=%.6f, L1Reg=%.6f, L2Reg=%.6f"
              % (baseline_loss, zero_loss, xavier_loss, he_loss, mse_loss, lrdecay_loss, l1_loss, l2_loss))
        print("(TrainAcc, TestAcc): Baseline=(%.6f, %.6f), ZeroInit=(%.6f, %.6f), XavierInit=(%.6f, %.6f), "
              "HeInit=(%.6f, %.6f), MSELoss=(%.6f, %.6f), LrDecay=(%.6f, %.6f), L1Reg=(%.6f, %.6f), L2Reg=(%.6f, %.6f)"
              % (baseline_train_acc, baseline_test_acc,
                 zero_train_acc, zero_test_acc,
                 xavier_train_acc, xavier_test_acc,
                 he_train_acc, he_test_acc,
                 mse_train_acc, mse_test_acc,
                 lrdecay_train_acc, lrdecay_test_acc,
                 l1_train_acc, l1_test_acc,
                 l2_train_acc, l2_test_acc))

        # 进行Tensorboard的绘制
        tb_writer.add_scalars("Loss",
                              {
                                  "Baseline": baseline_loss,
                                  "ZeroInit": zero_loss,
                                  "XavierInit": xavier_loss,
                                  "HeInit": he_loss,
                                  "MSELoss": mse_loss,
                                  "LrDecay": lrdecay_loss,
                                  "L1Reg": l1_loss,
                                  "L2Reg": l2_loss
                              },
                              (epoch+1))

        tb_writer.add_scalars("Accuracy",
                              {
                                  "Baseline-Train": baseline_train_acc,
                                  "Baseline-Test": baseline_test_acc,
                                  "ZeroInit-Train": zero_train_acc,
                                  "ZeroInit-Test": zero_test_acc,
                                  "XavierInit-Train": xavier_train_acc,
                                  "XavierInit-Test": xavier_test_acc,
                                  "HeInit-Train": he_train_acc,
                                  "HeInit-Test": he_test_acc,
                                  "MSELoss-Train": mse_train_acc,
                                  "MSELoss-Test": mse_test_acc,
                                  "LrDecay-Train": lrdecay_train_acc,
                                  "LrDecay-Test": lrdecay_test_acc,
                                  "L1Reg-Train": l1_train_acc,
                                  "L1Reg-Test": l1_test_acc,
                                  "L2Reg-Train": l2_train_acc,
                                  "L2Reg-Test": l2_test_acc
                              },
                              (epoch+1))

    return


if __name__ == '__main__':
    # print("===>  RUN")
    # run_baseline()
    compare_all_method()
