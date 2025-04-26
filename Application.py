import cv2

from Exceptions import UnsuccessfulCaptureException
from PipelineModules.Classificator.GraphTCN import GraphTcn
from PipelineModules.FeatureExtractor import FeatureExtractor, FeaturePackage
from PipelineModules.LmCapturer import LmCapturer
from PipelineModules.Classificator.WindowManager import WindowManager

def main() -> None:
    lm_capturer: LmCapturer = LmCapturer()
    feature_extractor: FeatureExtractor = FeatureExtractor()
    graph_tcn: GraphTcn = GraphTcn() #TODO fill in parameters
    frame_delta = 1/lm_capturer.cap.get(cv2.CAP_PROP_FPS)

    #Warm up until window is full
    while len(graph_tcn.getSeq()) < 30:
        cur_frame = lm_capturer.capture()
        if cur_frame is not None:  # DEBUG
            feature_package = feature_extractor.extract(cur_frame)
            graph_tcn.extractSpatialFeatures(feature_package.landmark_coordinates)


    #Recogniton Phase
    while True:
        try:
            cur_frame = lm_capturer.capture() #blocks until frame is read
            if cur_frame is not None: #DEBUG
                feature_package: FeaturePackage = feature_extractor.extract(cur_frame)
                print(str(feature_package.landmark_coordinates))
                #window_manager.update(feature_package)  windowmanager now included in the graph_tcn
                #todo recognize gestures and trigger events
        except UnsuccessfulCaptureException as e:
            print(e.message)
        cv2.waitKey(1) #DEBUG


if __name__ == '__main__':
    main()