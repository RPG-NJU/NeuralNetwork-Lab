import torch.nn as nn
from config import MLPConfig
from opData import *
from torch.utils.data import Dataset, DataLoader
from myModule import MyModule
import torch.optim as optim
from utils import sMAPE
import tensorboardX as tb


class SeqMLP(MyModule):
    def __init__(self, config=MLPConfig()):
        """
        初始化一个序列MLP，通过滑动窗口来进行MLP的训练，最终达到预测一个时间序列的目的
        :param config: Net Config
        """
        super(SeqMLP, self).__init__(config=config)

        # 设置网络全连接层和ReLU激活层
        self.fc_layer1 = nn.Linear(in_features=self.input_num, out_features=200, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc_layer2 = nn.Linear(in_features=200, out_features=100, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc_layer3 = nn.Linear(in_features=100, out_features=80, bias=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc_layer4 = nn.Linear(in_features=80, out_features=self.output_num, bias=True)

        # 设置损失函数
        self.loss_function = nn.MSELoss()
        # self.loss_function = nn.L1Loss()

        # 获得训练集/验证集/测试集
        self.validation_inorder_loader = DataLoader(MyDataset(self.train_data), shuffle=False, batch_size=self.train_data.shape[0])
        self.train_shuffle_loader = DataLoader(MyDataset(get_windows_data(self.train_data[:, :self.validation_index], self.input_num+self.output_num)), shuffle=True, batch_size=self.batch_size)
        self.test_inorder_loader = DataLoader(MyDataset(self.test_data), shuffle=False, batch_size=self.train_data.shape[0])

        # 设置优化器
        self.optimizer = optim.SGD(self.parameters(), lr=self.learn_rate, momentum=self.momentum)

    def forward(self, x):
        result = x
        result = self.fc_layer1(result)
        result = self.relu1(result)
        result = self.fc_layer2(result)
        result = self.relu2(result)
        result = self.fc_layer3(result)
        result = self.relu3(result)
        result = self.fc_layer4(result)
        return result

    def train_a_epoch(self):
        """
        进行一轮的训练。
        :return:
        """
        total_loss = 0.0
        total_n = 0.0

        for batch_data in self.train_shuffle_loader:
            self.optimizer.zero_grad()

            batch_train_data = batch_data[:, 0:self.input_num]  # 切分之前的INPUT_N个数据作为输入
            batch_target_data = batch_data[:, self.input_num:]  # 切分之后的OUTPUT_N个数据作为输出的对照
            # print(batch_train_data.shape)

            batch_output_data = self.forward(batch_train_data)

            loss = self.loss_function(batch_output_data, batch_target_data)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_output_data.shape[0]
            total_n += batch_train_data.shape[0]

        total_loss = total_loss / total_n
        return total_loss

    def validate(self):
        """
        进行验证
        :return:
        """
        with torch.no_grad():
            for val_data in self.validation_inorder_loader:
                val_input = val_data[:, self.validation_index-self.input_num: self.validation_index]
                val_target = val_data[:, self.validation_index:]
                val_output = self.forward(val_input)

                val_output, val_target = line_minmax_denorm(val_output, val_target, self.min, self.max)

                smape = sMAPE(val_output, val_target)
                return smape

    def test(self):
        """
        进行测试
        :return:
        """
        with torch.no_grad():
            train_inorder_iter = iter(self.validation_inorder_loader)
            test_inorder_iter = iter(self.test_inorder_loader)
            input_data = train_inorder_iter.__next__()[:, -self.input_num:]
            target_data = test_inorder_iter.__next__()

            output_data = self.forward(input_data)

            output_data, target_data = line_minmax_denorm(output_data, target_data, self.min, self.max)

            return sMAPE(output_data, target_data)


if __name__ == '__main__':
    mlp = SeqMLP(config=MLPConfig())
    tb_writer = tb.SummaryWriter(logdir=MLPConfig().TB_PATH)
    for epoch in range(0, MLPConfig().EPOCH):
        print("===>  Epoch: %d" % (epoch+1))
        train_loss = mlp.train_a_epoch()
        val_smape = mlp.validate()
        test_smape = mlp.test()
        print("Train Loss=%.6f, Validate sMAPE=%.3f, Test sMAPE=%.3f" % (train_loss, val_smape, test_smape))

        tb_writer.add_scalars(main_tag="Loss", tag_scalar_dict={
            "MLP-Train": train_loss
        }, global_step=epoch+1)
        tb_writer.add_scalars(main_tag="sMAPE", tag_scalar_dict={
            "MLP-Validate": val_smape,
            "MLP-Test": test_smape
        }, global_step=epoch+1)

