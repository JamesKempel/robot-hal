"""
what the robot controller itself says about whether it can move, for drivers that report it.

a UR stops by itself on a protective stop, a safeguard stop, an e-stop, a paused or stopped program, or a power off. The
trajectory controller then either stops reporting (it is deactivated) or holds its reference, and the motion executor would
only notice through the stale reference, the stall check or the finish deadline. The UR driver's io_and_status_controller
publishes the cause directly, URRobotStatus watches it and MotionExecutor(robot_status=...) asks it on every poll.

the driver publishes these only when they change, so a value that never arrived counts as no problem: the other checks
still catch the stop, just later. a value that did arrive and isn't fine stops the move.
"""
import threading

# ur_dashboard_msgs/msg/SafetyMode and RobotMode, the values a move can run in
_SAFETY_MODES = {
    1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY", 5: "SAFEGUARD_STOP", 6: "SYSTEM_EMERGENCY_STOP",
    7: "ROBOT_EMERGENCY_STOP", 8: "VIOLATION", 9: "FAULT", 10: "VALIDATE_JOINT_ID", 11: "UNDEFINED_SAFETY_MODE",
    12: "AUTOMATIC_MODE_SAFEGUARD_STOP", 13: "SYSTEM_THREE_POSITION_ENABLING_STOP",
}
_SAFETY_OK = {1, 2}
_ROBOT_MODES = {
    -1: "NO_CONTROLLER", 0: "DISCONNECTED", 1: "CONFIRM_SAFETY", 2: "BOOTING", 3: "POWER_OFF", 4: "POWER_ON", 5: "IDLE",
    6: "BACKDRIVE", 7: "RUNNING", 8: "UPDATING_FIRMWARE",
}
_ROBOT_RUNNING = 7

UR_STATUS_CONTROLLER = "io_and_status_controller"


class URRobotStatus:
    """subscribes to <namespace>/{safety_mode, robot_mode, robot_program_running}. the node must be spun in the background"""

    def __init__(self, node, namespace: str = f"/{UR_STATUS_CONTROLLER}"):
        # imported here so the library still imports without a sourced ROS env or without the UR packages
        from std_msgs.msg import Bool
        from ur_dashboard_msgs.msg import RobotMode, SafetyMode

        self._node = node
        self._lock = threading.Lock()
        self._safety_mode = self._robot_mode = self._program_running = None
        namespace = namespace.rstrip("/")
        self._subscriptions = [
            node.create_subscription(SafetyMode, f"{namespace}/safety_mode", self._on_safety_mode, 10),
            node.create_subscription(RobotMode, f"{namespace}/robot_mode", self._on_robot_mode, 10),
            node.create_subscription(Bool, f"{namespace}/robot_program_running", self._on_program_running, 10),
        ]

    def _on_safety_mode(self, msg):
        with self._lock:
            self._safety_mode = int(msg.mode)

    def _on_robot_mode(self, msg):
        with self._lock:
            self._robot_mode = int(msg.mode)

    def _on_program_running(self, msg):
        with self._lock:
            self._program_running = bool(msg.data)

    def problem(self) -> str | None:
        """why the robot can't follow a move right now, None when nothing says it can't"""
        with self._lock:
            safety, robot, running = self._safety_mode, self._robot_mode, self._program_running
        return status_problem(safety, robot, running)

    def close(self) -> None:
        for subscription in self._subscriptions:
            self._node.destroy_subscription(subscription)
        self._subscriptions = []


def status_problem(safety_mode: int | None, robot_mode: int | None, program_running: bool | None) -> str | None:
    """the UR status as a reason a move can't run, None for unknown values (nothing received yet) and for a robot that can"""
    if safety_mode is not None and safety_mode not in _SAFETY_OK:
        return f"safety mode {_SAFETY_MODES.get(safety_mode, safety_mode)}"
    if robot_mode is not None and robot_mode != _ROBOT_RUNNING:
        return f"robot mode {_ROBOT_MODES.get(robot_mode, robot_mode)} (not RUNNING: powered off, idle or booting)"
    if program_running is False:
        return "the External Control program isn't running (stopped or paused on the teach pendant)"
    return None
