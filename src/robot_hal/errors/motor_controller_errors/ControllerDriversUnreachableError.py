"""
triggers when trying to detect hardware and the controller drivers don't return a response within a set period of time.
hard stop error for the library.
"""

class ControllerDriversUnreachableError(Exception):

    def __init__(self, message, error_code):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} ({self.error_code})"

DRIVER_UNREACHABLE_ERROR = ControllerDriversUnreachableError("Controller drivers are unreachable", 401)