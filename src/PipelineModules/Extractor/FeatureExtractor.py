import threading
from typing import Callable

from src.Config import USE_CUSTOM_MP_MULTITHREADING, USE_FILTER
from src.PipelineModules.Extractor.ExtractStrat import ExtractStratFactory

"""
Extracts landmark-features from an image and captures them in a FeaturePackage.
Spatial features get extracted duríng the classification process as they also have to be trained in combination with the later tcn layer.
"""

class FeatureExtractor:
    def __init__(self,
                 getFrame: Callable[[], any],
                 start_event: threading.Event,
                 stop_event: threading.Event,
                 use_custom_mp_multithreading: bool = USE_CUSTOM_MP_MULTITHREADING,
                 useFilter: bool = USE_FILTER):

        self.extractStrat = ExtractStratFactory.get(getFrame, use_custom_mp_multithreading, start_event, stop_event, USE_FILTER)

    def spawn_thread(self):
        self.extractStrat.start()

    def get(self):
        return self.extractStrat.getNext()







