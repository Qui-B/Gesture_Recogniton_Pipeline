import threading
import torch
from typing import Callable, Any

from src.Config import INPUT_SIZE, NUM_OUTPUT_CLASSES, GCN_NUM_OUTPUT_CHANNELS, NUM_CHANNELS_LAYER1, \
 KERNEL_SIZE, DROPOUT, DEVICE, USE_CUSTOM_MP_MULTITHREADING
from src.PipelineModules.Classificator.GraphTCN import GraphTcn
from src.PipelineModules.FeatureExtractor import FeatureExtractor
from src.PipelineModules.FrameCapturer import FrameCapturer
from src.Utility.Dataclasses import SpatialFeaturePackage
from src.Utility.Enums import Gesture

class App:
    def __init__(self):
        self.stop_event = threading.Event()
        self.capturer: FrameCapturer = FrameCapturer(self.stop_event)
        self.extractor = FeatureExtractor(self.stop_event)
        self.classifier = GraphTcn(
            input_size=INPUT_SIZE,
            output_size=NUM_OUTPUT_CLASSES,
            gcn_output_channels=GCN_NUM_OUTPUT_CHANNELS,
            num_channels_layer1=NUM_CHANNELS_LAYER1,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT).to(DEVICE)
        # classificator.load_state_dict(torch.load("PipelineModules/Classificator/trained_weights.pth")) TODO UNCOMMENT!!!!!!
        print("GraphTcn initialization succeeded")

        fps = self.capturer.measure_camera_fps(90)
        print("Camera initialization succeeded FPS: " + str(fps))

    def warm_up(self, getFeaturePackage: Callable[[], Any]):
        print("Starting warm up...")
        while self.classifier.window.getLength() < 30:
            feature_package = getFeaturePackage()
            spatial_t = self.classifier.extractSpatialFeatures(feature_package.lm_coordinates)  # manually fill window to avoid already running the tcn
            spatial_feature_package: SpatialFeaturePackage = SpatialFeaturePackage(spatial_t, feature_package.hand_detected)
            self.classifier.window.update(spatial_feature_package)
        print("Warm up done")



    def startClassification(self, getFeaturePackage: Callable[[], Any]):
        print("Start classification")
        while not self.stop_event.is_set():
            feature_package = getFeaturePackage()
            with torch.no_grad():
                result_t: torch.Tensor = self.classifier(feature_package)
                result_val, result_class = torch.max(result_t, dim=1)
                result_class = Gesture(result_class.item())
                # print("Classification Result is " + result_class.name)
                # print("Tensor: " + str(result_t))
                # print("")


    def start(self):
        get_feature_package = None

        if USE_CUSTOM_MP_MULTITHREADING:
            get_feature_package = self.extractor.getNext
            extractor_thread = threading.Thread(target=self.extractor.run, args=(self.capturer.get,), daemon=True)
            extractor_thread.start()
        else:
            get_feature_package = lambda: self.extractor.extract(self.capturer.get())
        capture_thread = threading.Thread(target=self.capturer.run, daemon=True)
        capture_thread.start()
        self.warm_up(get_feature_package)
        self.startClassification(get_feature_package)



