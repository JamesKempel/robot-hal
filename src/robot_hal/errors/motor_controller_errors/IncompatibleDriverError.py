"""
If there are drivers that respond, but are not compatible with my calibration software.
We can currently support Torque, Position and Velocity only in version 0.0.1
"""


class IncompatibleDriverError(Exception):


    def __init__(self, message, error_code):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} ({self.error_code})"

INCOMPATIBLEDRIVER = IncompatibleDriverError("driver is incompatible with calibration software", 101)