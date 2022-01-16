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
import tensorboardX as tb


class Trainer:
    def __init__(self, config=LSTMConfig()):
        # 对Train和Test数据进行获取
        self.config = config
        self.input_size = config.INPUT_SIZE
        self.target_size = config.TARGET_SIZE
        self.target_seq_len = config.TARGET_SEQ_LEN
        self.batch_size = config.BATCH_SIZE

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
        elif self.config.LOSS_FUNCTION == "L1":
            self.loss_function = nn.L1Loss()
        else:
            print("No Such OPTIMIZER")
            exit(-1)

        # 优化器部分
        if self.config.OPTIMIZER == "SGD":
            self.optimizer = optim.SGD(self.net.parameters(), self.config.LEARN_RATE, self.config.MOMENTUM, weight_decay=self.config.WEIGHT_DECAY)
        elif self.config.OPTIMIZER == "ADAM":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.LEARN_RATE, weight_decay=self.config.WEIGHT_DECAY)
        else:
            print("No Such OPTIMIZER")
            exit(-1)

        # 生成Train的所有序列
        self.train_seq_windows = get_windows_data(self.train_seq_norm, self.input_size+self.target_size)

    def init_h_c_state(self, batch_size):
        h_0 = torch.zeros(self.config.LAYER_NUM, batch_size, self.config.HIDDEN_DIM)
        c_0 = torch.zeros(self.config.LAYER_NUM, batch_size, self.config.HIDDEN_DIM)
        # h_0 = torch.zeros(self.config.LAYER_NUM, 10, self.config.HIDDEN_DIM)
        # c_0 = torch.zeros(self.config.LAYER_NUM, 10, self.config.HIDDEN_DIM)
        return h_0, c_0

    def train_epoch(self):
        """
        进行一个Epoch的训练。
        :return:
        """
        self.net.train()

        # h_state, c_state = self.init_h_c_state()    # 每一次训练的时候，从头滑动，此时需要将H和C重新初始化
        # train_loader = DataLoader(
        #     SeqWindowDataset(self.train_seq_norm, input_size=self.input_size, target_size=self.target_size,
        #                      target_seq_len=self.target_size), batch_size=1, shuffle=False)
        train_loader = DataLoader(
            MyDataset(self.train_seq_windows),
            batch_size=self.batch_size,
            shuffle=True
        )
        epoch_loss = 0.0
        for index, data_pair in enumerate(train_loader):
            print("\rThis epoch training: %d/%d" % (index+1, len(train_loader)), end="")
            self.optimizer.zero_grad()
            input_data = data_pair[:, :self.input_size]
            target_data = data_pair[:, self.input_size:]
            h_state, c_state = self.init_h_c_state(batch_size=input_data.shape[0])
            input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
            target_data = target_data.reshape((target_data.shape[0], target_data.shape[1], 1))

            output_data, _, _ = self.net.forward(input_data, h_state, c_state)
            # print(output_data.shape, target_data.shape)
            loss = self.loss_function(output_data.permute(0, 2, 1), target_data)
            loss.backward()
            epoch_loss += loss.item()
            self.optimizer.step()
        print(end="\n")
        return epoch_loss / len(train_loader)

    def test(self, mode: str):
        self.net.eval()

        if mode == "validate":
            # seq = self.validate_seq_data
            seq_norm = self.validate_seq_norm
        else:
            # seq = self.test_seq_data
            seq_norm = self.test_seq_norm
        # seq是没有归一化的数据，seq_norm是归一化之后的数据

        # input_seq = seq_norm[:, :-self.target_seq_len]  # 作为输入的部分, (111, input_size)
        # gt_seq = seq_norm[:, -self.target_seq_len:]
        # output_seq = np.zeros((input_seq.shape[0], 0))
        # seq_len = input_seq.shape[1]
        # seq_num = input_seq.shape[0]

        with torch.no_grad():
            data_loader = DataLoader(SeqWindowDataset(seq_norm, input_size=self.input_size, target_size=self.target_size, target_seq_len=self.target_seq_len), batch_size=1, shuffle=False)
            data_iter = iter(data_loader)
            input_data, target_data = next(data_iter)
            input_data = input_data.permute(1, 2, 0)
            target_data = target_data.permute(1, 2, 0)

            h_state, c_state = self.init_h_c_state(111)
            target_shape = (seq_norm.shape[0], self.target_seq_len) # 最终输出的Shape

            output_data, _, _ = self.net.forward(input_data, h_state, c_state)
            loss = self.loss_function(output_data.permute(0, 2, 1), target_data)
            loss = loss.item()

            output_data = output_data.numpy()
            target_data = target_data.numpy()
            output_data = output_data.reshape(target_shape)
            target_data = target_data.reshape(target_shape)

            output_data = OpData.line_denorm(output_data, self.line_min, self.line_max)
            target_data = OpData.line_denorm(target_data, self.line_min, self.line_max)

            smape = utils.sMAPE(torch.from_numpy(output_data), torch.from_numpy(target_data))

            return loss, smape


if __name__ == '__main__':
    config = LSTMConfig()
    trainer = Trainer(config)
    tb_writer = tb.SummaryWriter(config.TB_PATH)
    for epoch in range(0, config.EPOCH):
        train_loss = trainer.train_epoch()
        validate_loss, validate_smape = trainer.test("validate")
        test_loss, test_smape = trainer.test("test")
        print("===>  Epoch: %d" % (epoch+1))
        print("Train Loss=%.6f, Validate Loss=%.6f, Validate sMAPE=%.3f, Test sMAPE=%.3f" % (train_loss, validate_loss, validate_smape, test_smape))

        # 绘制图像
        tb_writer.add_scalars("Loss",
                              {
                                  "LSTM-Best-Train": train_loss,
                                  "LSTM-Best-Val": validate_loss
                              }, epoch+1)

        tb_writer.add_scalars("sMAPE",
                              {
                                  "LSTM-Best-Val": validate_smape,
                                  "LSTM-Best-Test": test_smape
                              }, epoch + 1)


