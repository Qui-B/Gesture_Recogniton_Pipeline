import sys
import threading
import time

import cv2
import torch
from Config import INPUT_SIZE, NUM_OUTPUT_CLASSES, GCN_NUM_OUTPUT_CHANNELS, NUM_CHANNELS_LAYER1, NUM_CHANNELS_LAYER2, \
    KERNEL_SIZE, DROPOUT, DEVICE
from Utility.Dataclasses import SpatialFeaturePackage
from Utility.Enums import Gesture
from Utility.Exceptions import UnsuccessfulCaptureException
from PipelineModules.Classificator.GraphTCN import GraphTcn
from PipelineModules.FeatureExtractor import FeatureExtractor, FeaturePackage
from PipelineModules.FrameCapturer import FrameCapturer

def main() -> None:
        stop_event = threading.Event()

        frame_capturer: FrameCapturer = FrameCapturer(stop_event)
        fps = frame_capturer.measure_camera_fps(400)
        print("Camera initialization succeeded | FPS: " + str(fps))


        feature_extractor: FeatureExtractor = FeatureExtractor(stop_event)
        classificator: GraphTcn = GraphTcn(
            input_size=INPUT_SIZE,
            output_size=NUM_OUTPUT_CLASSES,
            gcn_output_channels=GCN_NUM_OUTPUT_CHANNELS,
            num_channels_layer1=NUM_CHANNELS_LAYER1,
            num_channels_layer2=NUM_CHANNELS_LAYER2,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT
        ).to(DEVICE)
        #classificator.load_state_dict(torch.load("PipelineModules/Classificator/trained_weights.pth")) TODO UNCOMMENT!!!!!!
        print("GraphTcn initialization succeeded")

        capture_thread = threading.Thread(target=frame_capturer.run, daemon=True)
        capture_thread.start()

        print("Staring warm up...")
        # Warm up until frame_deque is full
        while classificator.window.getLength() < 30:
            cur_frame = frame_capturer.get()
            feature_package = feature_extractor.extract(cur_frame)
            spatial_t = classificator.extractSpatialFeatures(feature_package.lm_coordinates) #manually fill window to avoid already running the tcn
            spatial_feature_package: SpatialFeaturePackage = SpatialFeaturePackage(spatial_t, feature_package.hand_detected)
            classificator.window.update(spatial_feature_package)
        print("Warm up done")
        # Recogniton Phase
        t0 = 0
        t1 = 0
        print("Starting classification...")
        print("Classification done")
        while True:
            try:
                cur_frame = frame_capturer.get()
                feature_package: FeaturePackage = feature_extractor.extract(cur_frame)
                with torch.no_grad():
                    result_t: torch.Tensor = classificator(feature_package)
                    result_val, result_class = torch.max(result_t, dim=1)
                    result_class = Gesture(result_class.item())
                    if result_class is not Gesture.Nothing and result_val > 0.9:
                        print("Classification Result is " + result_class.name)
                        print("Tensor: " + str(result_t))
                        print("")
            except UnsuccessfulCaptureException as e:
                print(e.message)

if __name__ == '__main__':
    main()