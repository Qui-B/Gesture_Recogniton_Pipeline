import unittest
from collections import deque

import cv2
import torch

from Config import INPUT_SIZE, NUM_OUTPUT_CLASSES, GCN_NUM_OUTPUT_CHANNELS, NUM_CHANNELS_LAYER1, NUM_CHANNELS_LAYER2, \
    KERNEL_SIZE, DROPOUT, SAMPLE_PICTURE_PATH, BATCH_SIZE, WINDOW_LENGTH, DEVICE
from PipelineModules.Classificator.GraphTCN import GraphTcn
from PipelineModules.DataClasses import SpatialFeaturePackage
from PipelineModules.FeatureExtractor import FeatureExtractor, FeaturePackage


class GraphTcnTest(unittest.TestCase):

    def __init__(self, methodName: str = "runTest"):
        super().__init__(methodName)

    @classmethod
    def setUpClass(cls):
        cls.graph_tcn = GraphTcn(
                                input_size = INPUT_SIZE,
                                output_size = NUM_OUTPUT_CLASSES,
                                gcn_output_channels=GCN_NUM_OUTPUT_CHANNELS,
                                num_channels_layer1 = NUM_CHANNELS_LAYER1,
                                num_channels_layer2 = NUM_CHANNELS_LAYER2,
                                kernel_size = KERNEL_SIZE,
                                dropout = DROPOUT).to(DEVICE)
        feature_extractor = FeatureExtractor()
        cls.feature_package: FeaturePackage = feature_extractor.extract(cv2.cvtColor(cv2.imread(SAMPLE_PICTURE_PATH), cv2.COLOR_BGR2RGB))
        spatial_t = cls.graph_tcn.extractSpatialFeatures(cls.feature_package.lm_coordinates)
        cls.spatial_feature_package: SpatialFeaturePackage = SpatialFeaturePackage(spatial_t, cls.feature_package.hand_detected)


    def test_getWindow_type(self):
        self.graph_tcn.window.update(self.spatial_feature_package)
        window = self.graph_tcn.window.getAsTensor()
        self.assertEqual(window.dtype, torch.float32)

    def test_getWindow_basic_dimensions(self):
        self.graph_tcn.window.clear()
        self.graph_tcn.window.update(self.spatial_feature_package)
        window = self.graph_tcn.window.getAsTensor()

        self.assertEqual(window.shape[0], BATCH_SIZE)
        self.assertEqual(window.shape[1], self.graph_tcn.window.FLATTENED_T_LENGTH)
        self.assertEqual(window.shape[2], 1)

    def test_updateWindow_time_dimension(self):
        self.graph_tcn.window.clear()
        self.graph_tcn.window.update(self.spatial_feature_package)
        cur_window = self.graph_tcn.window.getAsTensor()

        for i in range(1,WINDOW_LENGTH):
            self.assertEqual(i, cur_window.shape[2])
            self.graph_tcn.window.update(self.spatial_feature_package)
            cur_window = self.graph_tcn.window.getAsTensor()

    def test_updateWindow_tensor_pop(self):
        self.graph_tcn.window.clear()
        for i in range(0,WINDOW_LENGTH+1): #+1 to check frame_deque pop
            self.graph_tcn.window.update(self.spatial_feature_package)

        window = self.graph_tcn.window.getAsTensor()
        self.assertEqual(WINDOW_LENGTH,window.shape[2])

    def test_updateWindow_flattening(self):
        self.graph_tcn.window.update(self.spatial_feature_package)
        window = self.graph_tcn.window.getAsTensor()

        flattened_t = window[0, :, -1]
        multi_dim_t = self.feature_package.lm_coordinates #Dimension: [Landmarks, Coordinates]

        #compare hand detected elem
        self.assertEqual(self.feature_package.hand_detected, flattened_t[0])

        #compare landmarks
        x = 0
        y = 1
        z = 2
        for i,landmark in enumerate(multi_dim_t):
            self.assertAlmostEqual(landmark[x], flattened_t[3*i + 1]) #starting with +1 to skip the HAND_DETECTED_ELEM
            self.assertAlmostEqual(landmark[y], flattened_t[3*i + 2])
            self.assertAlmostEqual(landmark[z], flattened_t[3*i + 3])




