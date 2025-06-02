from src.Config import FRAMEWINDOW_LEN

class UnsuccessfulCaptureException(Exception):
    """
    Raised when the camera module could not capture a frame
    """
    def __init__(self, frame_number):
        self.message = "Camera module could not capture a frame, current frame_number: " + str(frame_number)
        self.frame_number = frame_number
        super().__init__(self.message)


class WindowLengthException(Exception): #TODO Maybe delete as not needed until now
    """
    Raised when the frame_deque-length differs from Setting.FRAME_WINDOW_LENGTH
    """
    def __init__(self, cur_window_length, video_name):
        self.cur_window_length = cur_window_length
        self.message = ("Window-length ({cur_length}) of Video: \"{video_name} \" differs from Setting.FRAME_WINDOW_LENGTH ({settings_length})"
                        .format(video_name=video_name,cur_length=cur_window_length, settings_length=FRAMEWINDOW_LEN))
        super().__init__(self.message)