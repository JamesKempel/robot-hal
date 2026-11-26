"""
errors raised when something could hurt the robot or quietly corrupt the calibration. All of them are hard stops.
The rule behind them: stopping with a descriptive error is always better than continuing and shipping a wrong model,
or moving the robot with formulas we can't trust.

stopped_safely tells the caller what state the robot was left in:
    True  -> the safety monitor stopped the robot (switched to the hold controller / deactivated the arm controller)
    False -> a stop was needed and it failed. the robot may still be moving, use the emergency stop
    None  -> raised before anything moved, or by a check that doesn't stop the robot itself
"""


class CalibrationSafetyError(Exception):

    def __init__(self, message, error_code, stopped_safely: bool | None = None):
        self.message = message
        self.error_code = error_code
        self.stopped_safely = stopped_safely
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} ({self.error_code})"


class TorqueDriftError(CalibrationSafetyError):
    """
    measured joint torque has drifted away from what the model predicts (or, on torque drivers, from what was commanded).
    on torque drivers the model computes every command, so a drifting model means the robot is being driven with wrong torques.
    on position drivers it means an unexpected load (wrong payload, a collision, a cable snag) whose deflection would get baked
    into the kinematic parameters.
    """


class TrackingError(CalibrationSafetyError):
    """the joints are not where they were commanded to be: stall, collision, saturated drive, or wrong dynamics on torque drivers"""


class JointLimitError(CalibrationSafetyError, ValueError):
    """a target or a measured joint state is outside the URDF limits. measured states outside them mean the URDF or encoders are wrong"""


class JointStateInvalidError(CalibrationSafetyError):
    """joint state feedback is missing, stale, frozen, NaN, or lacks something the safety checks need (e.g. effort on torque drivers)"""


class JointMappingError(CalibrationSafetyError):
    """the URDF's joints don't line up with the joints the robot reports, every FK prediction would use the wrong angles"""


class RobotNotStillError(CalibrationSafetyError):
    """the robot moved during a step that needs it still (sensor survey, measurement settle), the readings can't be trusted"""


class CalibrationAbortedError(CalibrationSafetyError):
    """another part of the library found something that invalidates the run (e.g. the sensor lost the marker) and stopped it"""


class ToolFramePlacementError(CalibrationSafetyError):
    """the URDF's tool0 and the controller's own tool0_controller disagree by more than the tolerance at the start pose: the
    URDF may not match the robot variant, or a joint has a zero offset too small for the workspace guard's clearance to catch"""


class SafeStopUnconfiguredError(CalibrationSafetyError):
    """there is no safe way to stop this robot configured, so nothing is allowed to move"""


class SafeStopFailedError(CalibrationSafetyError):
    """a stop was needed and it failed. the robot may still be moving or being driven, use the emergency stop"""

    def __init__(self, message, error_code):
        super().__init__(message, error_code, stopped_safely=False)


TORQUE_DRIFT = TorqueDriftError("measured joint torque drifted from the model", 301)
TRACKING_ERROR = TrackingError("joints are not following the commanded setpoint", 302)
JOINT_LIMIT = JointLimitError("joint state is outside the URDF limits", 303)
JOINT_STATE_INVALID = JointStateInvalidError("joint state feedback is missing, stale or invalid", 304)
JOINT_MAPPING = JointMappingError("URDF joints don't match the joints the robot reports", 305)
ROBOT_NOT_STILL = RobotNotStillError("robot moved while it had to be still", 306)
SAFE_STOP_UNCONFIGURED = SafeStopUnconfiguredError("no safe stop is configured for this robot", 307)
SAFE_STOP_FAILED = SafeStopFailedError("safe stop failed, use the emergency stop", 308)
CALIBRATION_ABORTED = CalibrationAbortedError("calibration was stopped because its results could not be trusted", 309)
TOOL_FRAME_PLACEMENT = ToolFramePlacementError("the URDF's tool0 and the controller's tool0_controller disagree", 310)
