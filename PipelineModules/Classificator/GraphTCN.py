
from torch import nn
import torch
from torch_geometric.graphgym import GCNConv

from HelperClasses import TemporalConvNet
from PipelineModules.Classificator.WindowManager import WindowManager
from PipelineModules.FeatureExtractor import FeaturePackage
from Settings import EDGE_INDEX


class GraphTcn(nn.Module):
    def __init__(self, input_size, output_size, num_channels_layer1, num_channels_layer2, kernel_size, dropout=0):
        super(GraphTcn, self).__init__()

        #for spatial features
        self.gcn1 = GCNConv(3, 12) #input 3 features per handpoint

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
        self.window_manager = WindowManager()
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def extractSpatialFeatures(self, landmark_coordinates):
        return self.gcn1(landmark_coordinates, EDGE_INDEX) #TODO abchecken welche dimension ich breauche

    def getWindowTensor(self):
        sequence_t = (torch.stack(list(self.window_manager.landmark_vector_deque))) #collect the whole window in a tensor
        sequence_t = sequence_t.unsqueeze(0) # to get a 3 dimensional vector with BATCH_SIZE  = 1
        return sequence_t

    def updateWindow(self, feature_package: FeaturePackage):
        spatial_t = self.extractSpatialFeatures(feature_package.landmark_coordinates)
        flattened_t = spatial_t.view(spatial_t.shape[0], -1) #flatten by concatenating columns
        window_t = torch.cat([feature_package.hand_detected_flag,flattened_t],dim=1) #append hand detected flag
        self.window_manager.update(window_t)

    def forward(self, feature_package: FeaturePackage): #x = landmark vector
        self.updateWindow(feature_package)
        sequence_t = self.getWindowTensor()


        #TODO ab hier beginnt die baustelle
        y0 = self.conv2d(x)
        B, C, H, W = y0.shape
        y0_flattened = y0.view(B, H * W, -1)
        y1 = self.tcn1(y0)
        y_pooled = self.pool(y1) #dim 2: max over time dimension
        y2 = self.tcn2(y_pooled)
        y_linear = self.linear(y2[:, :, -1])
        return self.activation_function(y_linear)


