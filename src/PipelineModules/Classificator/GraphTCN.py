from torch import torch,nn
from torch_geometric.nn.conv import GCNConv

from src.PipelineModules.Classificator.FrameWindow import FrameWindow
from src.Config import EDGE_INDEX, NUM_LANDMARK_DIMENSIONS
from src.PipelineModules.Classificator.HelperClasses import TemporalConvNet
from src.Utility.Dataclasses import FeaturePackage, SpatialFeaturePackage


class GraphTcn(nn.Module):
    def __init__(self, input_size, output_size, num_channels_layer1, gcn_output_channels,  kernel_size, dropout, feature_vector_width = NUM_LANDMARK_DIMENSIONS):
        super(GraphTcn, self).__init__()
        self.window = FrameWindow()

        self.gcn = GCNConv(feature_vector_width, gcn_output_channels)

        self.tcn = TemporalConvNet(
                input_size,
                num_channels_layer1,
                kernel_size,
                dropout)

        self.classifier = nn.Linear(num_channels_layer1[-1], output_size)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()




    def init_weights(self):
        self.classifier.weight.data.normal_(0, 0.01)


    def extractSpatialFeatures(self, landmark_coordinates):
        """
        returns a  21 * 8 feature tensor
        """
        x = self.gcn(landmark_coordinates, EDGE_INDEX)
        return torch.tanh(x)




    def forward(self, *feature_pkgs: FeaturePackage): #maybe add bool to decide for training or interference for more clarity and better runtime
        if len(feature_pkgs) == 1: #case inference
            feature_pkg = feature_pkgs[0]
            spatial_t = self.extractSpatialFeatures(feature_pkg.lm_coordinates)
            spatial_feature_pkg = SpatialFeaturePackage(spatial_t, feature_pkg.hand_detected)

            self.window.update(spatial_feature_pkg)
            window_t = self.window.getAsTensor()
            y0 = self.tcn(window_t)
            y_linear = self.classifier(y0[:, :, -1])
            return self.softmax(y_linear) #softmax output to allow a Confidence Setting

        else: #case: training
            for feature_pkg in feature_pkgs:
                #print(type(feature_pkg))
                spatial_t = self.extractSpatialFeatures(feature_pkg.lm_coordinates)
                spatial_feature_pkg = SpatialFeaturePackage(spatial_t, feature_pkg.hand_detected)

                self.window.update(spatial_feature_pkg)
            window_t = self.window.getAsTensor()
            y0 = self.tcn(window_t)
            return self.classifier(y0[:, :, -1]) #softmax happens later in the lossfunction (cross-entropy)