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

    def __init__(self, in_channels, out_channels, learn_rate=0.0001, weight_decay=0.0, reg_mode="None"):
        """
        初始化这一层全连接层
        :param in_channels: 输入的隐层or数据维度
        :param out_channels: 输出的隐层维度
        """
        super(FullyConnectLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learn_rate = learn_rate

        self.weight_decay = weight_decay
        self.reg_mode = reg_mode

        # 网络层的权重，shape=out*in，最终在运算阶段，采用W*X的形式
        self.weights = np.zeros((out_channels, in_channels))
        # Bias
        self.bias = np.zeros((out_channels, 1))

        self.x = None
        return

    def set_learn_rate(self, learn_rate):
        self.learn_rate = learn_rate

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
        delta_w = self.learn_rate * (loss @ self.x.T) / self.x.shape[1]
        delta_b = np.mean(self.learn_rate * loss, axis=1).reshape(self.out_channels, 1)

        if self.reg_mode != "None":
            # 如下进行正则化的传递
            if self.reg_mode == "L1":
                more_than_zero = np.where(self.weights > 0, 1, 0)
                less_than_zero = np.where(self.weights < 0, -1, 0)
                reg_matrix = self.weight_decay * (more_than_zero + less_than_zero)
                reg_matrix = reg_matrix * self.weights
            elif self.reg_mode == "L2":
                reg_matrix = self.weight_decay * self.weights
            else:
                exit(-1)
            delta_w += self.learn_rate * reg_matrix

        assert delta_w.shape == (self.out_channels, self.in_channels)
        assert delta_b.shape == (self.out_channels, 1)

        next_step_loss = self.weights.T @ loss
        self.weights -= delta_w
        self.bias -= delta_b

        # print(next_step_loss.shape)
        return next_step_loss


class ReLU(NNLayer):
    def __init__(self):
        super(ReLU, self).__init__()
        # self.weights = np.zeros((channels, 1))  # 用列向量来表示，实际运行的时候1应该是batch size
        # self.channels = channels
        self.weights = None

    def forward(self, x):
        self.weights = np.where(x > 0, 1, 0)
        assert self.weights.shape == x.shape

        return x * self.weights

    def backward(self, loss: np.ndarray):
        return loss * self.weights


class Sigmoid(NNLayer):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))

    def backward(self, loss):
        sigmoid = 1 / (1 + np.exp(-self.x))
        return sigmoid * (1 - sigmoid) * loss


class Softmax(NNLayer):
    def __init__(self):
        super(Softmax, self).__init__()
        self.y = None

    def forward(self, x):
        # print(x.shape, np.sum(np.exp(x), axis=0).shape)
        self.y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return self.y

    def backward(self, loss):
        assert loss.shape == (self.y.shape[1])
        next_step_loss = np.zeros(loss.shape)


class SoftmaxCELoss(NNLayer):
    def __init__(self):
        super(SoftmaxCELoss, self).__init__()
        self.y = None

    def forward(self, x, y):
        self.y = np.exp(x) / np.sum(np.exp(x), axis=0)
        self.y_one_hot = np.zeros(x.shape)
        for i in range(0, y.shape[0]):
            self.y_one_hot[y[i], i] = 1.0
        return np.sum(-self.y_one_hot * np.log(self.y)) / self.y.shape[1]

    def backward(self, loss):
        return self.y - self.y_one_hot


class CrossEntropyLoss(NNLayer):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, x: np.ndarray, y):
        assert y.shape[1] == 1
        y_one_hot = np.zeros(x.shape)
        for i in range(0, y.shape[0]):
            y_one_hot[y[i], i] = 1.0
        # 改成One Hot编码格式
        self.x = x
        self.y = y_one_hot
        return np.sum(-y * np.log(x))

    def backward(self, loss):
        return -self.y / self.x


class MSELoss(NNLayer):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.x = None
        self.y = None

    def forward(self, x: np.ndarray, y):
        """
        前向传播函数
        :param x: 网络的输出
        :param y: Labels，shape=(batch_size,)
        :return:
        """
        y_one_hot = np.zeros(x.shape)
        for i in range(0, y.shape[0]):
            y_one_hot[y[i], i] = 1.0
        # 改成One Hot编码格式
        self.x = x
        self.y = y_one_hot

        loss = y_one_hot - x
        loss = loss * loss
        loss = np.sum(loss)
        loss = loss / np.size(x)
        return loss

    def backward(self, loss):
        return 2 * (self.x - self.y)


if __name__ == '__main__':
    fc = FullyConnectLayer(4, 6)
    relu = ReLU()
    sigmoid = Sigmoid()
    x = [[3, 1, 2, 3], [2, 2, 3, 4], [3, 3, 2, 1], [4, 2, 3, 1]]
    x = np.array(x)
    x = x.T

    print(x.shape, np.sum(x, axis=1).shape)
    # x.reshape((3, 4))

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
    print(type(relu))
    print(type(relu) == ReLU)

    print(type(np.sum(np.sum(x, axis=1), axis=0)))

    softmax = Softmax()
    s = softmax.forward(x)
    print(s)

    print(x)
    y = [2, 3, 1, 2]
    y = np.array(y)
    print(y.shape)
    softmax_log = SoftmaxCELoss()
    ss = softmax_log.forward(x, y)
    print(ss)
    ss_loss = softmax_log.backward(ss)
    print(ss_loss)
