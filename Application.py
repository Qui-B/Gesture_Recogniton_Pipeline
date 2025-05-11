import cv2
import torch
from Config import INPUT_SIZE, NUM_OUTPUT_CLASSES, GCN_NUM_OUTPUT_CHANNELS, NUM_CHANNELS_LAYER1, NUM_CHANNELS_LAYER2, \
    KERNEL_SIZE, DROPOUT, DEVICE
from Enums import Gesture
from Exceptions import UnsuccessfulCaptureException
from PipelineModules.Classificator.GraphTCN import GraphTcn
from PipelineModules.DataClasses import SpatialFeaturePackage
from PipelineModules.FeatureExtractor import FeatureExtractor, FeaturePackage
from PipelineModules.LmCapturer import LmCapturer

def main() -> None:
        lm_capturer: LmCapturer = LmCapturer()
        feature_extractor: FeatureExtractor = FeatureExtractor()
        classificator: GraphTcn = GraphTcn(
            input_size=INPUT_SIZE,
            output_size=NUM_OUTPUT_CLASSES,
            gcn_output_channels=GCN_NUM_OUTPUT_CHANNELS,
            num_channels_layer1=NUM_CHANNELS_LAYER1,
            num_channels_layer2=NUM_CHANNELS_LAYER2,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT
        ).to(DEVICE)
        #classificator.load_state_dict(torch.load("trained_model.pth"))

        frame_delta = 1 / lm_capturer.cap.get(cv2.CAP_PROP_FPS)
        # Warm up until frame_deque is full
        while classificator.window.getLength() < 30:
            cur_frame = lm_capturer.capture()
            feature_package = feature_extractor.extract(cur_frame)
            spatial_t = classificator.extractSpatialFeatures(feature_package.lm_coordinates) #manually fill window to avoid already running the tcn
            spatial_feature_package: SpatialFeaturePackage = SpatialFeaturePackage(spatial_t, feature_package.hand_detected)
            classificator.window.update(spatial_feature_package)

        # Recogniton Phase
        while True:
            try:
                cur_frame = lm_capturer.capture()  # blocks until frame is read
                feature_package: FeaturePackage = feature_extractor.extract(cur_frame)
                with torch.no_grad():
                    result_t: torch.Tensor = classificator(feature_package)
                    result_class = Gesture(torch.argmax(result_t, dim=1).item())
                print("Classification Result is " + result_class.name + " Whole tensor: " + str(result_t))
            except UnsuccessfulCaptureException as e:
                print(e.message)
            cv2.waitKey(1)  # DEBUG

if __name__ == '__main__':
    main()