import threading

import win32file
import time
from abc import abstractmethod, ABC

import win32pipe

from src.Config import CONFIDENCE_THRESHOLD, GESTURE_COOLDOWN_S, PIPE_NAME, SEND_ACROBAT_EVENTS
from src.Utility.DebugManager.DebugManager import debug_manager



class EventHandlerFactory:
    @staticmethod
    def get(stop_event = None, send_events: bool = SEND_ACROBAT_EVENTS):
        return EventHandlerFactory.AcrobatEventHandler(stop_event) if send_events else EventHandlerFactory.NoEventHandler()

    class EventHandlerBase(ABC):
        def __init__(self, action_cooldown = GESTURE_COOLDOWN_S):
            self.action_cooldown = action_cooldown
            self.timestamp_last_action = 0  # temp val only present for the first scan

        @abstractmethod
        def handle(self, *args):
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
                     action_cooldown = GESTURE_COOLDOWN_S,
                     pipe_name = PIPE_NAME):
            super().__init__(action_cooldown)
            self.confidence_threshold = confidence_threshold

        def handle(self, confidence, gesture_event): #a bit wonky maybe change it so it does not depend on the conditionorder
            if (gesture_event != 0
                    and confidence > self.confidence_threshold
                        and self.cooldown_valid()):
                debug_manager.print_result(confidence, gesture_event)

    class AcrobatEventHandler(EventHandlerBase):
        stop_action = 0xFF #stop signal which gets received before the pipe is closed

        def __init__(self,
                     stop_event,
                     confidence_threshold = CONFIDENCE_THRESHOLD,
                     action_cooldown = GESTURE_COOLDOWN_S,
                     pipe_name = PIPE_NAME):

            super().__init__(action_cooldown)
            self.confidence_threshold = confidence_threshold
            self.stop_event = stop_event

            try:
                self.pipe = win32file.CreateFile(
                    pipe_name,
                    win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                    0,
                    None,
                    win32file.OPEN_EXISTING,
                    0,
                    None) #TODO handle via debughandler
            except Exception as e:
                print("ERROR: could not open pipe")

            print("thread started")
            self.stop_listener_thread = threading.Thread(
                target=self.listenForStopSignal
                )
            self.stop_listener_thread.start()


        def handle(self, confidence,gesture_event): #a bit wonky maybe change it so it does not depend on the conditionorder
            if (gesture_event != 0
                    and confidence >= self.confidence_threshold
                        and self.cooldown_valid()):
                try:
                    win32file.WriteFile(self.pipe, gesture_event.to_bytes(1, 'little'))
                    debug_manager.print_result(confidence, gesture_event)
                except Exception as e:
                    print("ERROR: could not write to pipe")


        def listenForStopSignal(self):
            print("thread running")
            while not self.stop_event.is_set():
                if not self.pipe:
                    print("No valid pipe handle. Listener exiting.")
                    break
                try:
                    success, n_bytes_avail, _ = win32pipe.PeekNamedPipe(self.pipe, 0)
                    if n_bytes_avail > 0:
                        error, data = win32file.ReadFile(self.pipe, 1)
                        if error != 0:
                            print("Stopping error. Shutting down listener...")
                            self.stop_event.set()
                            break
                        elif data[0] == self.stop_action:
                            print("Stop action received. Shutting down listener...")
                            self.stop_event.set()
                            time.sleep(0.1) #wait for mainloop to finish

                            win32file.WriteFile(self.pipe, self.stop_action.to_bytes(1, 'little'))
                            self.pipe.close()
                            break
                    time.sleep(0.1)  # reduced sleep for faster reaction
                except Exception as e:
                    print(f"Listener error: {e}")
                    break






