# 本文件中包括主要的LSTM网络结构
# 自定义一个网络结构
# 最终将该网络放到Trainer类中进行训练和测试
import torch
import torch.nn as nn
from config import LSTMConfig


class SeqLSTM(nn.Module):
    def __init__(self, config=LSTMConfig()):
        super(SeqLSTM, self).__init__()

        self.layer_num = config.LAYER_NUM
        self.hidden_dim = config.HIDDEN_DIM
        self.input_size = config.INPUT_SIZE
        self.target_size = config.TARGET_SIZE

        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_dim, num_layers=self.layer_num,
                            bias=True, batch_first=True, dropout=config.DROPOUT)
        # 需要注意，input size代表输入的特征数目，也就是特征维度，对于本次实验来说，输入的维度是1，因为只有一条数据

        # 需要加入一个线性层来得到一条数据

        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=100, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=100, out_features=self.target_size, bias=True)

    def forward(self, input, h, c):
        output, (h_next, c_next) = self.lstm(input, (h, c))
        output = output[:, -1:, :]
        # print(output.shape)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        return output, h_next, c_next


if __name__ == '__main__':
    lstm_config = LSTMConfig()

