import queue
import threading
import torch
import keyboard
import cv2

from src.Config import TCN_INPUT_SIZE, TCN_NUM_OUTPUT_CLASSES, GCN_NUM_OUTPUT_CHANNELS, TCN_CHANNELS, \
    KERNEL_SIZE, DROPOUT, DEVICE, USE_CUSTOM_MP_MULTITHREADING, WEIGHTS_FILE_PATH
from src.PipelineModules.Classificator.GraphTCN import GraphTcn
from src.PipelineModules.EventHandler import EventHandlerFactory
from src.PipelineModules.Extractor.FeatureExtractor import FeatureExtractor
from src.PipelineModules.FrameCapturer import FrameCapturer
from src.Utility.Dataclasses import SpatialFeaturePackage
from src.Utility.DebugManager.DebugManager import debug_manager


class App:
    def __init__(self):
        self.start_event = threading.Event()
        self.stop_event = threading.Event()

        self.capturer: FrameCapturer = FrameCapturer(self.start_event,self.stop_event)
        fps = self.capturer.measure_camera_fps(90)
        print(f"Capture-module initialization succeeded [FPS: {fps}, "
              f"Resolution: {int(self.capturer.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.capturer.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}]")

        self.extractor = FeatureExtractor(self.capturer.get, self.start_event, self.stop_event)
        print("Extractor-module initialization succeeded")

        self.classifier = GraphTcn(
            input_size=TCN_INPUT_SIZE,
            output_size=TCN_NUM_OUTPUT_CLASSES,
            gcn_output_channels=GCN_NUM_OUTPUT_CHANNELS,
            num_channels_layer1=TCN_CHANNELS,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT).to(DEVICE)
        self.classifier.load_state_dict(torch.load(WEIGHTS_FILE_PATH))
        print("Classifier-module initialization succeeded")

        self.event_handler = EventHandlerFactory.get()
        print("Eventhandler-module initialization succeeded")

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
                with torch.no_grad():
                    result_t: torch.Tensor = self.classifier(feature_package)
                    result_val, result_class = torch.max(result_t, dim=1)
                    self.event_handler.handle(result_val.item(), result_class.item())

                    # if result_class != Gesture.Nothing and result_val > 0.6: TODO delete one day...
                    #     print("Classification Result is " + result_class.name)
                    #     print("Tensor: " + str(result_t))
                    #     print("")
            except queue.Empty:
                debug_manager.log_framedrop("App.startClassification() -queue.Empty")
            debug_manager.print_framedrops()
            if keyboard.is_pressed('esc'):
                print("ESC pressed shutting down...")
                self.stop_event.set() #stops capturer and extractor
                self.event_handler.close()

                debug_manager.close()

        print("classifier stopped")


    def start(self):
        self.capturer.spawn_thread()
        print("Capture-module started")
        if USE_CUSTOM_MP_MULTITHREADING:
            self.extractor.spawn_thread()
            print("Extractor-module started")

        self.start_event.set()
        self.warm_up()
        self.startClassification()



