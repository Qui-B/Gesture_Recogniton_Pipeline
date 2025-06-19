from abc import ABC, abstractmethod
from src.Utility.Enums import Gesture


class DebugResultPrinterFactory:
    @staticmethod
    def get(print_results: bool):
        return DebugResultPrinter() if print_results else DebugNoResultPrinter



class DebugResultPrinterBase(ABC):
    @abstractmethod
    def print(self, *args):
        pass

class DebugNoResultPrinter(DebugResultPrinterBase):
    def print(self, *args):
        pass

class DebugResultPrinter(DebugResultPrinterBase):
    def print(self,  confidence: float, gesture: int):
        if gesture != 0: #filter out nothing
            gesture_name = Gesture(gesture).name
            print(f"Detected[{(confidence*100):.2f}%]: {gesture_name}")