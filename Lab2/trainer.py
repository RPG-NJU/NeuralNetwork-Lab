# 用于训练LSTM的训练器类别
# 及相关函数

from opData import *
from config import LSTMConfig
import numpy as np
from seqLSTM import SeqLSTM
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import utils


class Trainer:
    def __init__(self, config=LSTMConfig()):
        # 对Train和Test数据进行获取
        self.config = config
        self.input_size = config.INPUT_SIZE
        self.target_size = config.TARGET_SIZE
        self.target_seq_len = config.TARGET_SEQ_LEN

        # 从Data中分别读取Train和Test数据
        self.train_data = OpData.read_csv(config.TRAIN_DATA_PATH)
        self.test_data = OpData.read_csv(config.TEST_DATA_PATH)

        # 进行序列划分，得到Train，Validate和Test序列
        self.train_seq_data = self.train_data[:, :self.train_data.shape[1] - self.config.TARGET_SEQ_LEN]
        self.validate_seq_data = self.train_data[:, self.train_data.shape[1] - self.config.TARGET_SEQ_LEN - self.config.INPUT_SIZE:]
        self.test_seq_data = np.concatenate((self.train_data[:, -self.config.INPUT_SIZE:], self.test_data), axis=1)

        # 按照Train Seq来得到归一化使用的min和max
        self.line_min, self.line_max = OpData.get_norm_minmax(self.train_seq_data)

        # 得到归一化的数据
        self.train_seq_norm, _, _ = OpData.line_norm(self.train_seq_data, self.line_min, self.line_max)
        self.validate_seq_norm, _, _ = OpData.line_norm(self.validate_seq_data, self.line_min, self.line_max)
        self.test_seq_norm, _, _ = OpData.line_norm(self.test_seq_data, self.line_min, self.line_max)

        # 网络部分
        self.net = SeqLSTM(config=self.config)

        # 损失函数部分
        if self.config.LOSS_FUNCTION == "MSE":
            self.loss_function = nn.MSELoss()
        else:
            print("No Such OPTIMIZER")
            exit(-1)

        # 优化器部分
        if self.config.OPTIMIZER == "SGD":
            self.optimizer = optim.SGD(self.net.parameters(), self.config.LEARN_RATE, self.config.MOMENTUM)
        elif self.config.OPTIMIZER == "ADAM":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.LEARN_RATE)
        else:
            print("No Such OPTIMIZER")
            exit(-1)

    def init_h_c_state(self):
        h_0 = torch.zeros(self.config.LAYER_NUM, self.train_seq_norm.shape[0], self.config.HIDDEN_DIM)
        c_0 = torch.zeros(self.config.LAYER_NUM, self.train_seq_norm.shape[0], self.config.HIDDEN_DIM)
        # h_0 = torch.zeros(self.config.LAYER_NUM, 10, self.config.HIDDEN_DIM)
        # c_0 = torch.zeros(self.config.LAYER_NUM, 10, self.config.HIDDEN_DIM)
        return h_0, c_0

    def some_test(self):
        """
        在构建网络阶段用于测试的代码，最终不会再使用
        :return:
        """
        print("Some TEST!")
        loader = DataLoader(SeqWindowDataset(self.train_seq_norm, input_size=self.input_size, target_size=self.target_size, target_seq_len=self.target_size), batch_size=1, shuffle=False)
        h, c = self.init_h_c_state()
        total_loss = 0.0
        for data in loader:
            self.optimizer.zero_grad()
            h = h.detach()
            c = c.detach()
            input_data = data[0]
            target_data = data[1]
            # print(input_data.shape)
            # print(data[0].permute(1, 2, 0).shape, data[1].permute(1, 2, 0).shape)
            input_data = input_data.permute(1, 2, 0)
            target_data = target_data.permute(1, 2, 0)
            # [:, 0:10, :].
            # print(input_data.shape)
            output_data, h, c = self.net.forward(input_data, h, c)
            loss = self.loss_function(output_data[:, -self.target_size:, :], target_data)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
        print(total_loss / len(loader))

    def train_epoch(self):
        """
        进行一个Epoch的训练。
        :return:
        """
        h_state, c_state = self.init_h_c_state()    # 每一次训练的时候，从头滑动，此时需要将H和C重新初始化
        train_loader = DataLoader(
            SeqWindowDataset(self.train_seq_norm, input_size=self.input_size, target_size=self.target_size,
                             target_seq_len=self.target_size), batch_size=1, shuffle=False)
        epoch_loss = 0.0
        for data_pair in train_loader:
            self.optimizer.zero_grad()
            h_state = h_state.detach()
            c_state = c_state.detach()
            input_data = data_pair[0].permute(1, 2, 0)
            target_data = data_pair[1].permute(1, 2, 0)
            output_data, h_state, c_state = self.net.forward(input_data, h_state, c_state)
            loss = self.loss_function(output_data[:, -self.target_size:, :], target_data)
            loss.backward()
            epoch_loss += loss.item()
            self.optimizer.step()

        return epoch_loss / len(train_loader)

    def test(self, mode: str):
        if mode == "validate":
            seq = self.validate_seq_data
            seq_norm = self.validate_seq_norm
        else:
            seq = self.test_seq_data
            seq_norm = self.test_seq_norm
        # seq是没有归一化的数据，seq_norm是归一化之后的数据

        input_seq = seq_norm[:, :-self.target_seq_len]  # 作为输入的部分, (111, input_size)
        gt_seq = seq_norm[:, -self.target_seq_len:]
        output_seq = np.zeros((input_seq.shape[0], 0))
        seq_len = input_seq.shape[1]
        seq_num = input_seq.shape[0]

        with torch.no_grad():
            h_state, c_state = self.init_h_c_state()
            for i in range(0, self.target_seq_len):
                # if output_seq.shape[1] == 0
                if output_seq.shape[1] >= self.input_size:
                    net_input_seq = output_seq[:, -50:].reshape((seq_num, seq_len, 1)).astype(np.float32)
                else:
                    net_input_seq = np.concatenate((input_seq, output_seq), axis=1).reshape(
                        (seq_num, seq_len, 1)).astype(np.float32)
                net_input_seq_tensor = torch.from_numpy(net_input_seq)
                new_output_seq, h_state, c_state = self.net.forward(net_input_seq_tensor, h_state, c_state)
                new_output_seq = new_output_seq.numpy()[:, -1, 0].reshape((seq_num, 1))

                # 进行滑动
                if input_seq.shape[1] > 0:
                    input_seq = input_seq[:, 1:]
                output_seq = np.concatenate((output_seq, new_output_seq), axis=1)
                # print(input_seq.shape, output_seq.shape)

        assert output_seq.shape == gt_seq.shape

        with torch.no_grad():
            loss = self.loss_function(torch.from_numpy(output_seq.astype(np.float32)), torch.from_numpy(gt_seq.astype(np.float32)))
            smape = utils.sMAPE(torch.from_numpy(OpData.line_denorm(output_seq, self.line_min, self.line_max)), torch.from_numpy(OpData.line_denorm(gt_seq, self.line_min, self.line_min)))

        return loss.item(), smape


if __name__ == '__main__':
    trainer = Trainer()
    # print(trainer.train_seq_data.shape, trainer.validate_seq_data.shape, trainer.test_seq_data.shape)
    print(trainer.test("validate"))
    while True:
        print(trainer.train_epoch())
        print(trainer.test("validate"))

