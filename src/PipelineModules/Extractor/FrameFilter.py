from abc import ABC, abstractmethod

from src.Config import USE_FILTER, FILTER_CONSEC_FRAMEDROPS
from src.Utility.Dataclasses import FeaturePackage

class FrameFilterFactory:
    @staticmethod
    def get(use_filter: bool = USE_FILTER):
        return FrameFilterFactory.FrameFilter() if use_filter else FrameFilterFactory.NoFilter()

    class FilterBase(ABC):
        @abstractmethod
        def validate(self, input):
            pass

    class NoFilter(FilterBase):
        def validate(self,input):
            return input

    class FrameFilter(FilterBase):
        def __init__(self,drop_n_consec_frames: int = FILTER_CONSEC_FRAMEDROPS):
            self.max_consec_drops = drop_n_consec_frames
            self.n_consec_drops = 0

        def validate(self, feature_package: FeaturePackage):
            if feature_package.hand_detected:
                self.n_consec_drops = 0
                return True

            elif self.n_consec_drops == self.max_consec_drops:
                return True

            else: #case: no hand detected
                self.n_consec_drops += 1
                return False