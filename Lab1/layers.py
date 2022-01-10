# 实现了MLP中各层的具体结构
# 使用单独的文件以减少MLP.py文件中的代码行数，增强可读性

import numpy as np
# import math


class NNLayer:
    def __init__(self):
        return

    def forward(self, x):
        return

    def backward(self, loss):
        return


class FullyConnectLayer(NNLayer):

    def __init__(self, in_channels, out_channels, learn_rate=0.0001):
        """
        初始化这一层全连接层
        :param in_channels: 输入的隐层or数据维度
        :param out_channels: 输出的隐层维度
        """
        super(FullyConnectLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learn_rate = learn_rate

        # 网络层的权重，shape=out*in，最终在运算阶段，采用W*X的形式
        self.weights = np.zeros((out_channels, in_channels))
        # Bias
        self.bias = np.zeros((out_channels, 1))

        self.x = None
        return

    def forward(self, x: np.ndarray):
        """
        前向传播函数
        :param x: 输入的数据, shape=(in_channel, batch_size)
        :return: output Y, shape=(out_channel,batch_size)
        """
        self.x = x  # 记录当前输入的X

        return self.weights @ self.x + self.bias

    def backward(self, loss: np.ndarray):
        """
        反向传播函数
        :param loss: 上层传递的Loss值, shape=(out_channel, 1)
        :return: 继续传递的Loss值
        """
        x_mean = np.mean(self.x, axis=1)    # shape=(in_channels,)
        assert x_mean.shape == (self.in_channels, )

        delta_w = - self.learn_rate * (loss @ x_mean.reshape(self.in_channels, 1).T)
        delta_b = - self.learn_rate * loss
        assert delta_w.shape == (self.out_channels, self.in_channels)
        assert delta_b.shape == (self.out_channels, 1)

        next_step_loss = self.weights.T @ loss
        self.weights += delta_w
        self.bias += delta_b

        return next_step_loss


class ReLU(NNLayer):
    def __init__(self):
        super(ReLU, self).__init__()
        # self.weights = np.zeros((channels, 1))  # 用列向量来表示，实际运行的时候1应该是batch size
        # self.channels = channels
        self.weights = None

    def forward(self, x):
        # self.weights = np.zeros((self.channels, x.shape[1]))
        self.weights = np.where(x > 0, 1, 0)
        assert self.weights.shape == x.shape

        return x * self.weights

    def backward(self, loss: np.ndarray):
        w_mean = np.mean(self.weights, axis=1).reshape((self.channels, 1))

        return loss * w_mean


class Sigmoid(NNLayer):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, loss):
        sigmoid = 1 / (1 + np.exp(-loss))
        return sigmoid * (1 - sigmoid)


if __name__ == '__main__':
    fc = FullyConnectLayer(4, 6)
    relu = ReLU()
    sigmoid = Sigmoid()
    x = [[3, 1, 2, 3], [2, 2, 3, 4], [3, 3, 2, 1]]
    x = np.array(x)
    x = x.T

    print(x.shape)
    x.reshape((3, 4))

    y = fc.forward(x)
    print(y.shape)
    y[0, 1] = -4
    print(y)
    y_relu = relu.forward(y)
    y_sigmoid = sigmoid.forward(y)
    print(y_relu, y_sigmoid)
    loss = [[2], [4], [0], [0], [1], [5]]
    loss = np.array(loss)
    new_loss = fc.backward(loss)
    print(new_loss.shape)
