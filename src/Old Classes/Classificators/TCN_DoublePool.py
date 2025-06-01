
from torch import nn
import torch
from HelperClasses import TemporalConvNet


class TCNV2(nn.Module):
    def __init__(self, input_size, output_size, num_channels_layer1, num_channels_layer2, kernel_size, dropout=0):
        super(TCNV2, self).__init__()
        self.tcn1 = TemporalConvNet(input_size, num_channels_layer1, kernel_size=kernel_size, dropout=dropout)
        self.tcn2 = TemporalConvNet(num_channels_layer1[-1]/2, num_channels_layer2, kernel_size=kernel_size, dropout=dropout)
        self.tanh = nn.Tanh()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.linear = nn.Linear(num_channels_layer2[-1]/2, output_size) #numchannels: 8
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn1(x)#
        y_pooled = self.pool(y1)
        y2 = self.tcn2(y_pooled)
        y_pooled = self.pool(y2)
        y_linear = self.linear(y_pooled[:, :, -1])
        return self.tanh(y_linear)