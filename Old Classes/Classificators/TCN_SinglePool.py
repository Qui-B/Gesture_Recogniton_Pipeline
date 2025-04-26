
from torch import nn
import torch
from HelperClasses import TemporalConvNet


class TCN_SP(nn.Module):
    def __init__(self, input_size, output_size, num_channels_layer1, num_channels_layer2, kernel_size, dropout=0):
        super(TCN_SP, self).__init__()

        #for spatial features
        #Output dimension: 8*21*4
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)

        #First temporal layer
        #Output dimension: 8*21*4
        self.tcn1 = TemporalConvNet(
            8*21*4, #input size
            num_channels_layer1,
            kernel_size,
            dropout)

        self.tcn2 = TemporalConvNet(
            8*21*4, #same input size
            num_channels_layer2,
            kernel_size,
            dropout)

        self.activation_function = nn.Tanh()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.linear = nn.Linear(num_channels_layer2[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y0 = self.conv2d(x)
        B, C, H, W = y0.shape
        y0_flattened = y0.view(B, H * W, -1)
        y1 = self.tcn1(y0)
        y_pooled = self.pool(y1) #dim 2: max over time dimension
        y2 = self.tcn2(y_pooled)
        y_linear = self.linear(y2[:, :, -1])
        return self.activation_function(y_linear)