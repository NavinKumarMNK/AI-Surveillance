import torch
import torch.nn as nn
from collections import namedtuple
import lightning.pytorch as L

class MobileNetModel(nn.Module):
    def __init__(self, embedding_size):
        super(MobileNetModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        self.conv2_dw = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )

        self.conv_23 = self._depthwise_block(64, 128, 128, stride=(2, 2))
        self.conv_3 = self._residual_block(128, 128, 4, groups=128)

        self.conv_34 = self._depthwise_block(128, 256, 256, stride=(2, 2))
        self.conv_4 = self._residual_block(256, 256, 6, groups=256)

        self.conv_45 = self._depthwise_block(256, 128, 512, stride=(2, 2))
        self.conv_5 = self._residual_block(128, 512, 2, groups=256)

        self.conv_6 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512)
        )

        self.conv_6_dw = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), groups=512, bias=False),
            nn.BatchNorm2d(512)
        )

        self.conv_6_flatten = nn.Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        out = self.linear(out)
        out = self.bn(out)
        return self.l2_norm(out)

    def _depthwise_block(self, in_c, groups, out_c, stride):
        return nn.Sequential(
            nn.Conv2d(in_c, groups, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(groups),
            nn.PReLU(groups),
            nn.Conv2d(groups, groups, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups, bias=False),
            nn.BatchNorm2d(groups),
            nn.PReLU(groups),
            nn.Conv2d(groups, out_c, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        )

    def _residual_block(self, c, groups, num_block, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        modules = []
        for _ in range(num_block):
            modules.append(self._depthwise_block(c, groups, c, stride, kernel, padding))
        return nn.Sequential(*modules)

    def l2_norm(self, input, axis=1):
        norm = torch.norm(input, 2, axis, keepdim=True)
        output = torch.div(input, norm)
        return output


class MobileNet(L.LightningModule):
    def __init__(self, file_path=None, input_size=112):
        super(MobileNet, self).__init__()
        self.file_path = file_path
        self.example_input_array = torch.rand(1, 3, input_size, input_size)
        self.save_hyperparameters()
        
        if file_path:
            print("MobileNet weights loaded")
            self.model = torch.load(file_path + '.pt')
        else:        
            self.model = MobileNetModel(drop_ratio=0)
        
    def forward(self, x):
        return self.model(x)

    def save_model(self, file_path=None):
        print("Saving Model")
        self.file_path = file_path or self.file_path
        torch.save(self.model, self.file_path)
