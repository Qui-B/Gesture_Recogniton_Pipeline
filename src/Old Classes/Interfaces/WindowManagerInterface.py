from abc import ABC, abstractmethod

class WindowManagerInterface(ABC):
    @abstractmethod
    def getSeq(self):
        pass

    @abstractmethod
    def update(self, image):
        pass
