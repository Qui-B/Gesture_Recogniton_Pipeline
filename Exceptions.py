from Settings import WINDOW_LENGTH


class UnsuccessfulCaptureException(Exception):
    """
    Raised when the camera module could not capture a frame
    """
    def __init__(self, frame_number):
        self.message = "Camera module could not capture a frame, current frame_number: " + str(frame_number)
        self.frame_number = frame_number
        super().__init__(self.message)


class WindowLengthException(Exception):
    """
    Raised when the window-length differs from Setting.WINDOW_LENGTH
    """
    def __init__(self, cur_window_length):
        self.cur_window_length = cur_window_length
        self.message = "Window-length ({cur_length}) differs from Setting.WINDOW_LENGTH ({settings_length})".format(cur_length=cur_window_length, settings_length=WINDOW_LENGTH)
        super().__init__(self.message)
