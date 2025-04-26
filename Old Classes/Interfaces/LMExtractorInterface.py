from abc import ABC, abstractmethod

class LMExtractorInterface(ABC):
    @abstractmethod
    def capture(self):
        pass