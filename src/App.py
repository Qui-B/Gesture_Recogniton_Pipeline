import queue
import threading

import cv2
import torch
from typing import Callable, Any
import keyboard

from src.Config import INPUT_SIZE, NUM_OUTPUT_CLASSES, GCN_NUM_OUTPUT_CHANNELS, NUM_CHANNELS_LAYER1, \
    KERNEL_SIZE, DROPOUT, DEVICE, USE_CUSTOM_MP_MULTITHREADING, ESC
from src.PipelineModules.Classificator.GraphTCN import GraphTcn
from src.PipelineModules.FeatureExtractor import FeatureExtractor
from src.PipelineModules.FrameCapturer import FrameCapturer
from src.Utility.Dataclasses import SpatialFeaturePackage
from src.Utility.Enums import Gesture


class App:
    def __init__(self):
        self.start_event = threading.Event()
        self.stop_event = threading.Event()
        self.capturer: FrameCapturer = FrameCapturer(self.start_event,self.stop_event)
        print("main t0")
        fps = self.capturer.measure_camera_fps(90)
        print(f"Capture-module initialization succeeded [FPS: {fps}]")
        self.extractor = FeatureExtractor(self.capturer.get, self.start_event, self.stop_event)
        print("Extractor-module initialization succeeded")
        self.classifier = GraphTcn(
            input_size=INPUT_SIZE,
            output_size=NUM_OUTPUT_CLASSES,
            gcn_output_channels=GCN_NUM_OUTPUT_CHANNELS,
            num_channels_layer1=NUM_CHANNELS_LAYER1,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT).to(DEVICE)
        # classificator.load_state_dict(torch.load("PipelineModules/Classificator/trained_weights.pth")) TODO UNCOMMENT!!!!!!
        print("Classifier-module initialization succeeded")

    def warm_up(self):
        print("Starting warm up...")
        while self.classifier.window.getLength() < 30:
            feature_package = self.extractor.get()
            spatial_t = self.classifier.extractSpatialFeatures(feature_package.lm_coordinates)  # manually fill window to avoid already running the tcn
            spatial_feature_package: SpatialFeaturePackage = SpatialFeaturePackage(spatial_t, feature_package.hand_detected)
            self.classifier.window.update(spatial_feature_package)
        print("Warm up done")

    def startClassification(self):
        print("Start classification")
        while not self.stop_event.is_set():
            try:
                feature_package = self.extractor.get()  # throws Queue.Empty
            except queue.Empty as e:
                continue
            with torch.no_grad():
                result_t: torch.Tensor = self.classifier(feature_package)
                result_val, result_class = torch.max(result_t, dim=1)
                result_class = Gesture(result_class.item())
                # print("Classification Result is " + result_class.name)
                # print("Tensor: " + str(result_t))
                # print("")
            if keyboard.is_pressed('esc'):
                self.extractor.debugManager.close()
                print("ESC pressed shutting down...")
                self.stop_event.set()

        print("classifier stopped")


    def start(self):
        self.capturer.start()
        print("capturer started")
        if USE_CUSTOM_MP_MULTITHREADING:
            self.extractor.start()
            print("extractor started")

        self.start_event.set()
        self.warm_up()
        self.startClassification()



