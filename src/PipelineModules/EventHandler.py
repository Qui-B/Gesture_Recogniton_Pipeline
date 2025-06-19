import time
from abc import abstractmethod, ABC

from src.Config import CONFIDENCE_THRESHOLD, ACTION_COOLDOWN_S, PIPE_NAME, SEND_EVENTS
from src.Utility.DebugManager.DebugManager import debug_manager

class EventHandlerFactory:
    @staticmethod
    def get(send_events: bool = SEND_EVENTS):
        return EventHandlerFactory.AcrobatEventHandler() if send_events else EventHandlerFactory.NoEventHandler()

    class EventHandlerBase(ABC):
        def __init__(self,action_cooldown = ACTION_COOLDOWN_S):
            self.action_cooldown = action_cooldown
            self.timestamp_last_action = 0  # temp val only present for the first scan

        @abstractmethod
        def handle(self, *args):
            pass

        @abstractmethod
        def close(self):
            pass

        def cooldown_valid(self):
            timestamp_cur = time.time()
            time_dif = timestamp_cur - self.timestamp_last_action
            if time_dif > self.action_cooldown:
                self.timestamp_last_action = timestamp_cur
                return True
            else:
                return False

    class NoEventHandler(EventHandlerBase):
        def __init__(self,
                     confidence_threshold = CONFIDENCE_THRESHOLD,
                     action_cooldown = ACTION_COOLDOWN_S,
                     pipe_name = PIPE_NAME):
            super().__init__(action_cooldown)
            self.confidence_threshold = confidence_threshold

        def handle(self, confidence, gesture_event): #a bit wonky maybe change it so it does not depend on the conditionorder
            if (gesture_event != 0
                    and self.cooldown_valid()
                    and confidence > self.confidence_threshold):
                debug_manager.print_result(confidence, gesture_event)

        def close(self):
            pass

    class AcrobatEventHandler(EventHandlerBase):
        def __init__(self,
                     confidence_threshold = CONFIDENCE_THRESHOLD,
                     action_cooldown = ACTION_COOLDOWN_S,
                     pipe_name = PIPE_NAME):
            super().__init__(action_cooldown)
            self.pipe = open(pipe_name, "wb", buffering=0)
            self.confidence_threshold = confidence_threshold

        def handle(self, confidence,gesture_event): #a bit wonky maybe change it so it does not depend on the conditionorder
            if (gesture_event != 0
                    and self.cooldown_valid()
                    and confidence > self.confidence_threshold):
                self.pipe.write(gesture_event.to_bytes(1, 'little'))
                debug_manager.print_result(confidence, gesture_event)

        def close(self):
            self.pipe.close()
