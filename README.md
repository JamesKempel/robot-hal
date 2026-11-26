# robot_hal

A hardware abstraction layer for ROS 2 arms: detect what driver a controller actually speaks, discover external
precision sensors on the network and over USB, and move the arm with a safety monitor watching every joint state on
the way. It's a thin layer on top of `ros2_control`, not a replacement for it — detection goes through the controller
manager's `list_controllers`, and motion goes through the arm's own `FollowJointTrajectory` controller.

**Status:** everything here is usable standalone with nothing beyond a running `ros2_control` setup —
`HardwareControllerInterface` (driver detection) and `sensor_discovery` (external sensor auto-detection) need no
extra code at all, and `MotionExecutor`/`SafetyMonitor` need a URDF loaded through `robot_hal.parser.Parser`, shipped
in this package. `Parser` covers a single arm bolted to the cell with single-axis joints — the common case. If your
robot doesn't fit that (a non-fixed base, a multi-DOF joint), see **The Parser protocol** below for what to implement
instead.

## What's in it

| module | what it does |
| --- | --- |
| `hardware_communication.hardware_controller_interface` | Queries `ros2_control`'s controller manager and classifies the active controllers as `TORQUE` or `POS_VELO` (`DriverType`). Everything else branches on this. |
| `hardware_communication.sensor_discovery` | Finds GigE Vision cameras, Qualisys and OptiTrack motion capture servers (UDP broadcast discovery) and USB cameras (sysfs), and says what's still needed to turn each into a pose source. |
| `hardware_communication.motion_executor` | Plans a joint-space move (quintic, rest to rest, under a fraction of the URDF's velocity/acceleration limits), sends it as one `FollowJointTrajectory` goal, and watches the controller's own reported reference the whole way. |
| `hardware_communication.safety_monitor` | Checks every joint state for stale/NaN feedback, limit violations, tracking error and torque drift; stops the robot the moment any of them trips, through a `ControllerSafeStop` that switches controllers via `ros2_control`. |
| `hardware_communication.background_executor` | Spins an `rclpy` node on a background thread so subscriptions and action futures keep updating while the caller's thread plans, waits and checks. |
| `hardware_communication.ros_context` | Starts `rclpy` without its own signal handlers, so `SIGINT`/`SIGTERM` raise in the main thread instead of tearing down the context — the safe stop needs a working context to reach the controller manager. |
| `hardware_communication.robot_status` | Watches a UR driver's `io_and_status_controller` for safety/robot mode and program state, so `MotionExecutor` notices a protective stop, pause or power-off as soon as the driver reports it, not just through the finish deadline. |
| `hardware_communication.tool_frame_check` | Compares the URDF's `tool0` against the controller's own `tool0_controller` frame at the start pose, to catch a wrong robot variant or a joint zero offset before the first move. |
| `parser` | Builds a Pinocchio model from a URDF (fixed base, single-axis joints) and the per-driver-type command formula `MotionExecutor`/`SafetyMonitor` need — the reference implementation of **The Parser protocol** below. |
| `errors` | The shared error vocabulary: `motor_controller_errors` (driver detection/compatibility) and `safety_errors` (everything `SafetyMonitor` and `MotionExecutor` can raise). |

## Setup

Requires Python 3.10+ (ROS 2 Humble's version, the oldest LTS still in production use).

```sh
uv sync
uv run pytest
```

`rclpy` and the ROS message packages (`control_msgs`, `sensor_msgs`, `trajectory_msgs`, `ur_dashboard_msgs`, ...) are
not declared as dependencies: on a robot they come from the sourced ROS install, and a pip-installed `rclpy` built
for a different ROS distro would shadow it. Every module that needs them imports lazily, so the package imports fine
without a sourced ROS environment — only the ROS-touching calls themselves need one. On a machine with no ROS
install at all (to run the pure-Python parts, or the test suite's non-ROS tests), the `ros` extra installs `rclpy`
from PyPI: `uv sync --extra ros`.

## Quickstart

### Detect what driver the robot is running

This works the moment `ros2_control` is up — nothing else to implement:

```python
from robot_hal import HardwareControllerInterface

hardware = HardwareControllerInterface(joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                                                     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"])
print(hardware.get_driver_type)        # DriverType.POS_VELO or DriverType.TORQUE
print(hardware.joint_controllers())    # {"scaled_joint_trajectory_controller": ["shoulder_pan_joint/position", ...]}
```

`joint_names` is optional; pass it (your URDF's actuated joints) to ignore controllers claiming interfaces on other
joints (grippers, a second arm on the same `controller_manager`).

### Discover an external precision sensor

Also standalone — no robot connection needed:

```python
from robot_hal.hardware_communication.sensor_discovery import discover_devices

for device in discover_devices():
    print(device.describe())
    # "Basler acA2500 [GigE Vision] at 192.168.1.50: camera found but nothing turns its images into poses. ..."
```

This tells you a device is there and what's still missing (a fiducial, a pose estimator, a relay to
`PoseStamped`/TF) — it doesn't itself produce poses.

### Move the robot, watched

`MotionExecutor` and `SafetyMonitor` are driver-agnostic — they work against `DriverType.TORQUE` or
`DriverType.POS_VELO` the same way — but both need a `parser`: an object wrapping your robot's URDF and
[Pinocchio](https://github.com/stack-of-tasks/pinocchio) model (see **The Parser protocol** below for exactly what
that is; `robot_hal.parser.Parser` is the reference implementation, used here):

```python
import rclpy
from robot_hal import BackgroundExecutor, HardwareControllerInterface, MotionExecutor
from robot_hal.hardware_communication.motion_executor import JointTrajectoryTransport
from robot_hal.hardware_communication.ros_context import init_rclpy
from robot_hal.hardware_communication.safety_monitor import JointStateListener, SafetyMonitor, build_safe_stop
from robot_hal.parser import Parser

parser = Parser("my_robot.urdf")                            # add base_pose= for a wall- or ceiling-mounted arm

init_rclpy()                                                 # starts rclpy without its own signal handlers
node = rclpy.create_node("robot_hal_example")

hardware = HardwareControllerInterface(node, joint_names=parser.joint_names)
driver_type = hardware.get_driver_type                       # DriverType.POS_VELO or DriverType.TORQUE
parser.generate_formulas(driver_type)                         # tip_frame= to measure somewhere other than tool0/flange

safe_stop = build_safe_stop(hardware, driver_type)            # works out how this robot can be stopped
safe_stop.connect()

with BackgroundExecutor(node):
    joint_states = JointStateListener(node, parser.joint_names)
    monitor = SafetyMonitor(parser, safe_stop)
    transport = JointTrajectoryTransport(node, "scaled_joint_trajectory_controller", parser.joint_names)
    executor = MotionExecutor(parser, monitor, transport, joint_states)

    result = executor.move_to(goal_positions)   # blocks until the goal is reached, or raises
```

`move_to` raises a `safety_errors` subclass (never lets a trip pass silently) the moment the monitor sees something
wrong — a joint limit, a tracking error, torque drift, stale feedback, or the robot reporting its own stop — and
stops the robot first, before the exception propagates.

## The Parser protocol

`SafetyMonitor` and `MotionExecutor` need an object (called `parser` in both) that wraps a URDF-derived
[Pinocchio](https://github.com/stack-of-tasks/pinocchio) model and a per-driver-type command formula:

```python
from typing import Protocol
import numpy as np
import pinocchio as pin
from robot_hal.hardware_communication.hardware_controller_interface import DriverType


class Formulas(Protocol):
    driver_type: DriverType

    def command(self, positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray) -> np.ndarray:
        """builds (and range-checks) the command for this driver type from a planned joint-space point."""

    def forward_kinematics(self, positions: np.ndarray, frame: str) -> pin.SE3:
        """the named frame's pose at this joint configuration."""


class Parser(Protocol):
    model: pin.Model                 # needs model.createData() for RNEA (SafetyMonitor's torque-drift check)
    joint_names: list[str]           # URDF joint order — every array below is in this order
    position_limits: tuple[np.ndarray, np.ndarray]   # (lower, upper), radians
    velocity_limits: np.ndarray      # radians/s, 0.0 where the URDF leaves a joint unset
    effort_limits: np.ndarray        # N·m
    formulas: Formulas

    def to_model_configuration(self, positions: np.ndarray) -> np.ndarray:
        """URDF joint order -> Pinocchio's model configuration vector (q)."""

    def to_model_tangent(self, values: np.ndarray, kind: str = "velocity") -> np.ndarray:
        """URDF joint order -> Pinocchio's tangent space (v or a); kind is "velocity" or "acceleration"."""

    def joint_part(self, model_vector: np.ndarray) -> np.ndarray:
        """the inverse of to_model_tangent: Pinocchio's tangent space -> URDF joint order."""
```

`robot_hal.parser.Parser` implements exactly this, for a single arm bolted to the cell with single-axis joints
(revolute, continuous, prismatic) — the common case for a ROS 2 arm, and probably all you need. Write your own only
if your robot doesn't fit that: a non-fixed base (legs, wheels, thrusters), or a multi-DOF joint `/joint_states`
can't report as one value per joint. For a simple 6R arm, `to_model_configuration`/`to_model_tangent`/`joint_part`
are the identity (Pinocchio's model order already matches the URDF's), and the main work is `formulas.command`
range-checking against `effort_limits` or `position_limits`/`velocity_limits` depending on `driver_type` — see
`parser.py`'s `_build_formulas` for a worked example.

## Errors

Every error `robot_hal` raises comes from `robot_hal.errors` and is either a `motor_controller_errors`
(`ControllerDriversUnreachableError`, `IncompatibleDriverError`) or a `safety_errors` subclass of
`CalibrationSafetyError` (`JointLimitError`, `TrackingError`, `TorqueDriftError`, `JointStateInvalidError`,
`JointMappingError`, `RobotNotStillError`, `SafeStopUnconfiguredError`, `SafeStopFailedError`,
`CalibrationAbortedError`, `ToolFramePlacementError`). Catch `CalibrationSafetyError` to handle any safety trip the
same way; the message always says what was observed and why it tripped.

## License

[Apache License 2.0](LICENSE).

## Freedom commitments

This is the open layer of a larger project that also sells a paid calibration product built on top of it
(this isn't that — if you're looking for automatic per-unit calibration, that's a separate, not-yet-public package).
For this package specifically:

- Features are never removed from here to be sold back as paid.
- All tests for everything in this package are in this repository.
- No artificial limits — performance or otherwise — are built into this code.

## Testing

```sh
uv run pytest    # pure-Python + fake-transport tests, no ROS install or robot needed
```
