from torch import torch,nn
from torch_geometric.nn.conv import GCNConv

from src.PipelineModules.Classificator.FrameWindow import FrameWindow
from src.Config import EDGE_INDEX, NUM_LANDMARK_DIMENSIONS, DEVICE, DROPOUT
from src.PipelineModules.Classificator.HelperClasses import TemporalConvNet
from src.Utility.Dataclasses import FeaturePackage, SpatialFeaturePackage


class GraphTcn(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 num_channels_layer1,
                 gcn_output_channels,
                 kernel_size,
                 dropout = DROPOUT, #tcn dropout
                 feature_vector_width = NUM_LANDMARK_DIMENSIONS,
                 gcn_dropout = DROPOUT):
        super(GraphTcn, self).__init__()
        self.window = FrameWindow()

        self.gcn_dropout = nn.Dropout(gcn_dropout) #drop out after the gcn before the tcn
        self.gcn = GCNConv(feature_vector_width, gcn_output_channels)

        self.tcn = TemporalConvNet(
                input_size,
                num_channels_layer1,
                kernel_size,
                dropout)

        self.classifier = nn.Linear(num_channels_layer1[-1], output_size)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

        self.last_t = torch.zeros((21, 3), device=DEVICE)

    def init_weights(self):
        self.classifier.weight.data.normal_(0, 0.01)


    def extractSpatialFeatures(self, landmark_coordinates):
        """
        returns a  21 * 8 feature tensor
        """
        x = self.gcn(landmark_coordinates, EDGE_INDEX)
        x_activation = torch.tanh(x)
        return self.gcn_dropout(x_activation)

    def normalize(self, input_t: torch.Tensor):
        relative_t = input_t - self.last_t
        self.last_t = input_t #CAREFULl no Copy
        return relative_t

    def forward(self, *feature_pkgs: FeaturePackage): #maybe add bool to decide for training or interference for more clarity and better runtime
        if len(feature_pkgs) == 1: #case inference
            feature_pkg = feature_pkgs[0]
            normalized_t = self.normalize(feature_pkg.lm_coordinates)
            spatial_t = self.extractSpatialFeatures(normalized_t)
            spatial_feature_pkg = SpatialFeaturePackage(spatial_t, feature_pkg.hand_detected)

            self.window.update(spatial_feature_pkg)
            window_t = self.window.getAsTensor()
            y0 = self.tcn(window_t)
            y_linear = self.classifier(y0[:, :, -1])
            return self.softmax(y_linear)

        else: #case: training
            self.window.clear() #not directly needed, only for safety measures to guarantee no crossinterference between windows
            for feature_pkg in feature_pkgs:
                #print(type(feature_pkg))
                normalized_t = self.normalize(feature_pkg.lm_coordinates)
                spatial_t = self.extractSpatialFeatures(normalized_t)
                spatial_feature_pkg = SpatialFeaturePackage(spatial_t, feature_pkg.hand_detected)

                self.window.update(spatial_feature_pkg)
            self.last_t = torch.zeros((21, 3), device=DEVICE) #reset last_t to avoid cross interference between samples
            window_t = self.window.getAsTensor()
            y0 = self.tcn(window_t)
            return self.classifier(y0[:, :, -1]) #softmax happens later in the lossfunction (cross-entropy)