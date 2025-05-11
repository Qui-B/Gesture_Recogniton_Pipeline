from typing import Any

from torch import torch,nn
from torch_geometric.nn.conv import GCNConv

from PipelineModules.Classificator.HelperClasses import TemporalConvNet
from PipelineModules.Classificator.FrameWindow import FrameWindow
from PipelineModules.DataClasses import SpatialFeaturePackage, FeaturePackage
from Config import EDGE_INDEX, FEATURE_VECTOR_WIDTH, FEATURE_VECTOR_LENGTH, POOLSTRIDE


class GraphTcn(nn.Module):
    def __init__(self, input_size, output_size, num_channels_layer1, num_channels_layer2, gcn_output_channels ,  kernel_size, dropout):
        super(GraphTcn, self).__init__()
        self.window = FrameWindow()

        self.gcn = GCNConv(FEATURE_VECTOR_WIDTH, gcn_output_channels)

        self.tcn = nn.Sequential(
            TemporalConvNet(
                input_size,
                num_channels_layer1,
                kernel_size,
                dropout),
            nn.MaxPool1d(kernel_size=2, stride=POOLSTRIDE),
            TemporalConvNet(
                int(num_channels_layer1[-1] / POOLSTRIDE),
                num_channels_layer2,
                kernel_size,
                dropout)
        )

        self.classifier = nn.Linear(num_channels_layer2[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.classifier.weight.data.normal_(0, 0.01)


    def extractSpatialFeatures(self, landmark_coordinates):
        """
        returns a  21 * 8 feature tensor
        """
        return self.gcn(landmark_coordinates, EDGE_INDEX)




    def forward(self, *feature_pkgs): #maybe add bool for more clarity and better runtime
        if len(feature_pkgs) == 1: #case inference
            spatial_t = self.extractSpatialFeatures(feature_pkgs[0].lm_coordinates)
            spatial_feature_pkg = SpatialFeaturePackage(spatial_t, feature_pkgs[0].hand_detected)

            self.window.update(spatial_feature_pkg)
            window_t = self.window.getAsTensor()

        else: #case: training
            for feature_pkg in feature_pkgs:
                spatial_t = self.extractSpatialFeatures(feature_pkg)
                spatial_feature_pkg = SpatialFeaturePackage(spatial_t, feature_pkg.hand_detected)

                self.window.update(spatial_feature_pkg)
            window_t = self.window.getAsTensor()

        y0 = self.tcn(window_t)
        y_linear = self.classifier(y0[:, :, -1])
        predicted_class = torch.argmax(y_linear, dim=1)
        return predicted_class


    #Old implementation was to split both methods as it is more performant, but not as reliable in terms of trainingsusability.
    """
    #forward used for training. All spatial features get calculated on the spot to allow better backpropagation through the gcn.
    #The forward methods are not combined to avoid un-necessary overhead during inference
    def forward(self, feature_pkg_list: [FeaturePackage]):
        for feature_pkg in feature_pkg_list:
            spatial_t = self.extractSpatialFeatures(feature_pkg)
            spatial_feature_pkg = SpatialFeaturePackage(spatial_t, feature_pkg.hand_detected)
            self.window.update(spatial_feature_pkg)
        window_t = self.window.getAsTensor()

        y0 = self.tcn1(window_t)
        y_pooled = self.pool(y0)
        y1 = self.tcn2(y_pooled)

        y_linear = self.classifier(y1[:, :, -1])
        return self.softmax(y_linear)
    #forward used for inference. Only newest spatial feature gets calculated. Others get reused fromm previous invocations. (frame-window)
    def forward(self, feature_pkg: FeaturePackage):
        spatial_t = self.extractSpatialFeatures(feature_pkg.lm_coordinates)
        spatial_feature_pkg = SpatialFeaturePackage(spatial_t, feature_pkg.hand_detected)

        self.window.update(spatial_feature_pkg)
        window_t = self.window.getAsTensor()

        y0 = self.tcn1(window_t)
        y_pooled = self.pool(y0)
        y1 = self.tcn2(y_pooled)

        y_linear = self.classifier(y1[:, :, -1])
        return self.softmax(y_linear)
    """