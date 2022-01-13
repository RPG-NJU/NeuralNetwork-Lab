# 负责整个MLP的网络结构
from layers import NNLayer, FullyConnectLayer, ReLU, Sigmoid, MSELoss, SoftmaxCELoss
from enum import Enum
from config import Config
import numpy as np
from opData import OpData
import utils


class InitMethod(Enum):
    ZERO = 1
    UNIFORM = 2
    GAUSS = 3
    XAVIER = 4
    HE = 5


class MLP(NNLayer):
    def __init__(self, config=Config()):
        super(MLP, self).__init__()

        self.layer_list = []

        self.learn_rate = config.INIT_LEARN_RATE
        self.batch_size = config.BATCH_SIZE

        self.loss_function = MSELoss()

        self.train_images = OpData.read_idx3_file(config.TRAIN_IMAGES_PATH)
        self.train_labels = OpData.read_idx1_file(config.TRAIN_LABELS_PATH).astype(int)
        self.test_images = OpData.read_idx3_file(config.TEST_IMAGES_PATH)
        self.test_labels = OpData.read_idx1_file(config.TEST_LABELS_PATH).astype(int)

        return

    def forward(self, x):
        result = x
        for i in range(0, len(self.layer_list)):
            result = self.layer_list[i].forward(result)
        return result

    def backward(self, loss):
        next_step_loss = loss
        for i in range(1, len(self.layer_list) + 1):
            next_step_loss = self.layer_list[-i].backward(next_step_loss)
            # print(next_step_loss.shape)
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
                    layer.weights = np.random.normal(loc=0, scale=args[0] ** 0.5, size=weights_num).reshape(weights_shape)
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

    def train(self):
        assert self.train_labels.shape[0] % self.batch_size == 0
        assert self.test_labels.shape[0] % self.batch_size == 0
        # batch size必须可以整除
        batch_num = int(self.train_images.shape[0] / self.batch_size)

        total_loss = 0.0
        total_acc = 0.0

        for i in range(0, batch_num):
            x = self.train_images[i*self.batch_size: (i+1)*self.batch_size].T
            y_gt = self.train_labels[i*self.batch_size: (i+1)*self.batch_size]

            y = self.forward(x)
            loss = self.loss_function.forward(y, y_gt)
            total_loss += loss
            total_acc += utils.labels_equal_num(utils.y_to_labels(y), y_gt)
            back_loss = self.loss_function.backward(loss)
            self.backward(back_loss)

        total_loss = total_loss / batch_num
        total_acc = total_acc / self.train_labels.shape[0]

        return total_loss, total_acc

    def test(self):
        batch_num = int(self.train_images.shape[0] / self.batch_size)

        total_acc = 0.0

        for i in range(0, batch_num):
            x = self.test_images[i*self.batch_size: (i+1)*self.batch_size].T
            y_gt = self.test_labels[i*self.batch_size: (i+1)*self.batch_size]

            y = self.forward(x)
            total_acc += utils.labels_equal_num(utils.y_to_labels(y), y_gt)

        total_acc = total_acc / self.test_labels.shape[0]

        return total_acc


class BaselineMLP(MLP):
    def __init__(self, config=Config()):
        super(BaselineMLP, self).__init__(config=config)
        self.fc1 = FullyConnectLayer(784, 100, self.learn_rate)
        self.fc2 = FullyConnectLayer(100, 50, self.learn_rate)
        self.fc3 = FullyConnectLayer(50, 10, self.learn_rate)
        self.layer_list = [
            self.fc1,
            ReLU(),
            self.fc2,
            ReLU(),
            self.fc3,
        ]

        self.loss_function = SoftmaxCELoss()

        self.init(method=InitMethod.UNIFORM, args=[-0.1, 0.1])


if __name__ == '__main__':
    print("MLP.py")




