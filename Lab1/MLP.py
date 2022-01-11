# 负责整个MLP的网络结构
from layers import NNLayer, FullyConnectLayer, ReLU, Sigmoid
from enum import Enum
import random
import numpy as np


class InitMethod(Enum):
    ZERO = 1
    UNIFORM = 2
    GAUSS = 3
    XAVIER = 4
    HE = 5


class MLP(NNLayer):
    def __init__(self):
        super(MLP, self).__init__()

        self.layer_list = []

        return

    def forward(self, x):
        result = x
        for i in range(0, len(self.layer_list)):
            result = self.layer_list[i].forward(result)
        return result

    def backward(self, loss):
        next_step_loss = loss
        for i in range(1, len(self.layer_list) + 1):
            next_step_loss = self.layer_list[-i].backward(loss)
        return next_step_loss

    def init(self, method: InitMethod, args: list):
        """
        初始化网络
        :param method: 初始化的方法
        :param args: 参数列表，例如均匀分布的范围等: [low, high] for uniform, [variance] for gauss,
        :return: None
        """
        if method == InitMethod.ZERO:
            for layer in self.layer_list:
                if type(layer) == FullyConnectLayer:
                    layer.weights = 0
                    layer.bias = 0
        elif method == InitMethod.UNIFORM:
            assert len(args) == 2
            for layer in self.layer_list:
                if type(layer) == FullyConnectLayer:
                    weights_shape = layer.weights.shape
                    weights_num = np.size(layer.weights)
                    layer.weights = np.random.uniform(args[0], args[1], weights_num).reshape(weights_shape)
                    layer.bias = 0
        elif method == InitMethod.GAUSS:
            assert len(args) == 1
            for layer in self.layer_list:
                if type(layer) == FullyConnectLayer:
                    weights_shape = layer.weights.shape
                    weights_num = np.size(layer.weights)
                    layer.weights - np.random.normal(loc=0, scale=args[0] ** 0.5, size=weights_num).reshape(weights_shape)
                    layer.bias = 0
        elif method == InitMethod.XAVIER:
            all_fc_layer = []
            for layer in self.layer_list:
                if type(layer) == FullyConnectLayer:
                    all_fc_layer.append(layer)
            assert len(all_fc_layer) != 0
            for i in range(0, len(all_fc_layer)):
                if i - 1 >= 0:
                    m_last = np.size(all_fc_layer[i-1].weights)
                else:
                    m_last = 0
                m = np.size(all_fc_layer[i].weights)
                weights_shape = all_fc_layer[i].weights.shape

                r = 4 * ((6 / m_last + m) ** 0.5)
                all_fc_layer[i].weights = np.random.uniform(-r, r, m).reshape(weights_shape)
                all_fc_layer[i].bias = 0
        elif method == InitMethod.HE:
            for layer in self.layer_list:
                if type(layer) == FullyConnectLayer:
                    weights_shape = layer.weights.shape
                    weights_num = np.size(layer.weights)
                    r = (6 / weights_num) ** 0.5
                    layer.weights = np.random.uniform(-r, r, weights_num).reshape(weights_shape)
                    layer.bias = 0
        else:
            print("Error: No Such Init Method!")
            exit(-1)


class BaselineMLP(MLP):
    def __init__(self):
        super(BaselineMLP, self).__init__()
        self.fc1 = FullyConnectLayer(784, 100)
        self.fc2 = FullyConnectLayer(100, 50)
        self.fc3 = FullyConnectLayer(50, 10)
        self.layer_list = [
            self.fc1,
            ReLU(),
            self.fc2,
            ReLU(),
            self.fc3,
        ]


if __name__ == '__main__':
    print("MLP.py")




