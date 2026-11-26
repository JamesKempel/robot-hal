"""
watches the robot while the library moves it and stops it the moment anything could hurt it or corrupt the calibration.

why stop instead of correcting: a feedback loop that fixes torque drift has to run at the drive's rate with hard real time
guarantees, which ros2 topics in python can't give (and mvp.md §9 keeps our own control loop out of scope). A late or wrong
correction is worse than none, so the monitor detects, stops, and explains. Fixing the model is a human decision.

what it checks on every joint state (check()):
    feedback       stale, NaN, wrong size, or no effort on torque drivers         -> JointStateInvalidError  (immediate)
    limits         measured position outside the URDF                            -> JointLimitError         (immediate)
                   measured velocity over the URDF limit, or the planned move's cap -> JointLimitError         (debounced)
    tracking       joints too far from the setpoint they are following           -> TrackingError           (debounced)
    torque drift   measured effort vs the model's RNEA, and vs the torque command -> TorqueDriftError        (debounced)
and assert_still() for steps that need the robot still                          -> RobotNotStillError      (immediate)
is_still() asks the same question without tripping, for waiting out the settling after a move.

debounced checks need max_consecutive_violations readings in a row, so a single noisy sample doesn't stop a run.
the first trip latches: the robot is stopped through safe_stop, and every later check() raises the same error until a new
monitor is made. If safe_stop itself fails, SafeStopFailedError is raised instead and says to use the emergency stop.

how the robot is stopped (build_safe_stop / ControllerSafeStop): one strict ros2_control switch that deactivates the arm's
controllers and activates a hold controller. Deactivating alone is only allowed for position controllers, which keep holding
their last command. An effort controller that is just switched off stops producing holding torque and the arm can drop, so
torque robots must name a hold controller (e.g. a joint_trajectory_controller with effort PID gains) or nothing moves.
a refused switch is checked against list_controllers: a driver that already switched the arm controller off (UR's
controller_stopper) has done the stop's work, only what is still undone is retried and only a real refusal is a failed stop.
"""
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pinocchio as pin

from ..errors.safety_errors import (
    CalibrationAbortedError,
    CALIBRATION_ABORTED,
    CalibrationSafetyError,
    JointLimitError,
    JOINT_LIMIT,
    JointMappingError,
    JOINT_MAPPING,
    JointStateInvalidError,
    JOINT_STATE_INVALID,
    RobotNotStillError,
    ROBOT_NOT_STILL,
    SafeStopFailedError,
    SAFE_STOP_FAILED,
    SafeStopUnconfiguredError,
    SAFE_STOP_UNCONFIGURED,
    TorqueDriftError,
    TORQUE_DRIFT,
    TrackingError,
    TRACKING_ERROR,
)
from .hardware_controller_interface import DriverType, HardwareControllerInterface


@dataclass(frozen=True)
class JointState:
    """one joint state reading, in the URDF's joint order. received_at is time.monotonic() when it arrived"""
    received_at: float
    positions: np.ndarray
    velocities: np.ndarray | None = None
    efforts: np.ndarray | None = None


def joint_state_from_msg(msg, joint_names, received_at: float) -> JointState:
    """
    converts a sensor_msgs/JointState into a JointState in joint_names order. /joint_states can list joints in any order and
    include joints we don't care about (grippers), so every joint is looked up by name.
    """
    names = list(msg.name)
    missing = [joint for joint in joint_names if joint not in names]
    if missing:
        raise JointMappingError(
            f"/joint_states doesn't report the URDF joints {missing}, it reports {names}. check the URDF and the driver use "
            "the same joint names (tf_prefix)",
            JOINT_MAPPING.error_code,
        )
    order = [names.index(joint) for joint in joint_names]

    def pick(values, field):
        values = list(values)
        if not values:
            return None
        if len(values) != len(names):
            raise JointStateInvalidError(
                f"/joint_states {field} has {len(values)} entries for {len(names)} joints", JOINT_STATE_INVALID.error_code
            )
        return np.array([values[i] for i in order], dtype=float)

    positions = pick(msg.position, "position")
    if positions is None:
        raise JointStateInvalidError("/joint_states has no positions", JOINT_STATE_INVALID.error_code)
    return JointState(received_at, positions, pick(msg.velocity, "velocity"), pick(msg.effort, "effort"))


# a joint without a URDF velocity limit is assumed to move no faster than this, for the jump check (rad/s)
_UNSET_VELOCITY_LIMIT = 6.0
# encoder noise and quantization between two readings (rad)
_JUMP_SLACK_RAD = 0.005


class SafetyMonitor:
    def __init__(self, parser, safe_stop: Callable[[str], None], effort_tolerance_fraction: float = 0.08,
                 effort_tolerance_floor: float = 5.0, max_tracking_error_rad: float = 0.05, max_consecutive_violations: int = 3,
                 max_state_age_sec: float = 0.1, still_velocity_rad_s: float = 0.002, position_limit_margin_rad: float = 0.01,
                 velocity_limit_margin: float = 0.05, check_effort: bool | None = None, clock: Callable[[], float] = time.monotonic):
        """
        parser: URDFParser with generate_initial_formulas() already called, its model and driver type are what's checked against.
        safe_stop: called with the reason when the monitor trips, must stop the robot or raise. see build_safe_stop().
        effort_tolerance_fraction / effort_tolerance_floor: a joint's torque has drifted when it is off by more than
            max(floor, fraction * the joint's URDF effort limit), in Nm. the defaults are starting points, not validated on
            hardware: static friction alone moves measured torque off the model, so record the drift on a healthy run and set
            the tolerance just above it.
        max_tracking_error_rad: how far a joint may be from its setpoint.
        max_consecutive_violations: readings in a row a drift/tracking violation has to last before tripping.
        max_state_age_sec: older joint states count as stale feedback.
        still_velocity_rad_s: fastest a joint may move and still count as still.
        position_limit_margin_rad / velocity_limit_margin: slack on the URDF limits for measured states (absolute rad / fraction).
        check_effort: compare measured effort against the model. defaults to on for torque drivers only, since position drivers
            often report effort in other units (e.g. motor current), which would trip falsely. torque drivers can't run without it.
        clock: time source, same clock as JointState.received_at.
        """
        if safe_stop is None:
            raise SafeStopUnconfiguredError(
                "a safety monitor needs a safe_stop, nothing may move without a way to stop it", SAFE_STOP_UNCONFIGURED.error_code
            )
        self._parser = parser
        self._model = parser.model
        # own workspace, the monitor may run on a different thread than the babble's FK
        self._data = parser.model.createData()
        self._joint_names = parser.joint_names
        self._n_joints = len(self._joint_names)
        self._driver_type = parser.formulas.driver_type
        self._safe_stop = safe_stop
        self._effort_tolerance = np.maximum(effort_tolerance_floor, effort_tolerance_fraction * parser.effort_limits)
        self._max_tracking_error_rad = max_tracking_error_rad
        self._max_consecutive_violations = max_consecutive_violations
        self._max_state_age_sec = max_state_age_sec
        self._still_velocity_rad_s = still_velocity_rad_s
        self._position_limit_margin_rad = position_limit_margin_rad
        self._velocity_limit_margin = velocity_limit_margin
        self._check_effort = self._driver_type == DriverType.TORQUE if check_effort is None else check_effort
        self._clock = clock
        self._violations = {"tracking": 0, "drift": 0, "command": 0, "speed": 0, "velocity_limit": 0}
        self._tripped: CalibrationSafetyError | None = None
        # the last reading checked, for the jump check: two readings can't be further apart than the joints can move
        self._previous: JointState | None = None

    @property
    def max_state_age_sec(self) -> float:
        """older joint states count as stale feedback"""
        return self._max_state_age_sec

    @property
    def tripped(self) -> CalibrationSafetyError | None:
        """the error that stopped the robot, None while everything is fine"""
        return self._tripped

    def check(self, state: JointState, setpoint=None, acceleration=None, command=None, velocity_cap=None) -> None:
        """
        state: the latest joint state.
        setpoint: the position the joints are following right now (the interpolated trajectory point, not the final goal).
        acceleration: the setpoint's acceleration, used for the model torque. zeros if None, fine for slow babble motion.
        command: the JointCommand sent for this setpoint. on torque drivers its efforts are checked against the measured effort.
        velocity_cap: per joint, the fastest the current move may go (rad/s). the URDF limit alone allows many times the
            planned speed, this catches a controller running a different trajectory than the one that was checked.
        """
        self._raise_if_tripped()
        self._check_feedback(state)
        self._check_measured_limits(state)

        if velocity_cap is not None and state.velocities is not None:
            cap = np.asarray(velocity_cap, dtype=float)
            fast = self._names_where(np.abs(state.velocities) > cap)
            self._debounce("speed", fast, lambda: JointLimitError(
                f"joints {fast} are moving faster than the planned move allows (up to "
                f"{np.max(np.abs(state.velocities) / np.maximum(cap, 1e-9)):.1f}x the cap) for "
                f"{self._max_consecutive_violations} readings in a row. the controller is not running the checked trajectory",
                JOINT_LIMIT.error_code,
            ))

        if setpoint is not None:
            error = np.abs(state.positions - np.asarray(setpoint, dtype=float))
            far = self._names_where(error > self._max_tracking_error_rad)
            self._debounce("tracking", far, lambda: TrackingError(
                f"joints {far} are up to {np.max(error):.4f}rad from their setpoint (limit {self._max_tracking_error_rad}rad) for "
                f"{self._max_consecutive_violations} readings in a row. likely a collision, stall or saturated drive"
                + (", or the dynamics model is wrong and the torque commands don't produce the planned motion"
                   if self._driver_type == DriverType.TORQUE else ""),
                TRACKING_ERROR.error_code,
            ))

        if not self._check_effort:
            return
        if self._parser.model is not self._model:
            # adopt_model() swapped in a corrected model, predict torque from it from now on
            self._model = self._parser.model
            self._data = self._model.createData()
        velocities = state.velocities if state.velocities is not None else np.zeros(self._n_joints)
        acceleration = np.zeros(self._n_joints) if acceleration is None else np.asarray(acceleration, dtype=float)
        parser = self._parser
        predicted = parser.joint_part(pin.rnea(
            self._model, self._data, parser.to_model_configuration(state.positions), parser.to_model_tangent(velocities),
            parser.to_model_tangent(acceleration, "acceleration"),
        ))
        drift = np.abs(state.efforts - predicted)
        drifted = self._names_where(drift > self._effort_tolerance)
        self._debounce("drift", drifted, lambda: TorqueDriftError(
            f"measured torque on {drifted} is off the model by up to {self._worst(drift)} for "
            f"{self._max_consecutive_violations} readings in a row. "
            + ("the dynamics model the torque commands are computed from no longer matches the robot (wrong payload or "
               "inertias in the URDF, a collision, friction), continuing would drive it with wrong torques"
               if self._driver_type == DriverType.TORQUE else
               "something is loading the arm (payload not in the URDF, collision, cable snag). its deflection would be "
               "calibrated into the kinematic parameters"),
            TORQUE_DRIFT.error_code,
        ))

        if command is not None and command.efforts is not None:
            gap = np.abs(state.efforts - np.asarray(command.efforts, dtype=float))
            undelivered = self._names_where(gap > self._effort_tolerance)
            self._debounce("command", undelivered, lambda: TorqueDriftError(
                f"drives on {undelivered} are not delivering the commanded torque, off by up to {self._worst(gap)} for "
                f"{self._max_consecutive_violations} readings in a row. drive saturation, fault or limiting",
                TORQUE_DRIFT.error_code,
            ))

    def is_still(self, state: JointState) -> bool:
        """
        whether the robot is still, without tripping when it isn't. for waiting out the settling after a move: the babble polls
        this until it holds for the dwell time, then calls assert_still() for the measurement. bad feedback still trips.
        """
        return not self._moving_joints(state)

    def assert_still(self, state: JointState) -> None:
        """for steps that need a motionless robot, e.g. the sensor survey or a measurement once is_still() has held"""
        moving = self._moving_joints(state)
        if moving:
            self._trip(RobotNotStillError(
                f"joints {moving} are moving (up to {np.max(np.abs(state.velocities)):.4f}rad/s, still means under "
                f"{self._still_velocity_rad_s}rad/s) during a step that needs the robot still. the readings taken now would "
                "measure the motion, not the robot",
                ROBOT_NOT_STILL.error_code,
            ))

    def stop(self, reason: str) -> None:
        """stops the robot for a reason found outside the monitor (e.g. the external sensor lost the marker mid babble)"""
        self._raise_if_tripped()
        self._trip(CalibrationAbortedError(reason, CALIBRATION_ABORTED.error_code))

    def _moving_joints(self, state: JointState) -> list[str]:
        self._raise_if_tripped()
        self._check_feedback(state)
        if state.velocities is None:
            self._trip(JointStateInvalidError(
                "/joint_states has no velocities, can't confirm the robot is still", JOINT_STATE_INVALID.error_code
            ))
        return self._names_where(np.abs(state.velocities) > self._still_velocity_rad_s)

    def _check_feedback(self, state: JointState):
        age = self._clock() - state.received_at
        if age > self._max_state_age_sec:
            self._trip(JointStateInvalidError(
                f"latest joint state is {age:.3f}s old (limit {self._max_state_age_sec}s), the robot can't be watched without "
                "fresh feedback",
                JOINT_STATE_INVALID.error_code,
            ))
        for field, values in (("positions", state.positions), ("velocities", state.velocities), ("efforts", state.efforts)):
            if values is None:
                continue
            if np.shape(values) != (self._n_joints,):
                self._trip(JointStateInvalidError(
                    f"joint state {field} has shape {np.shape(values)}, the URDF has {self._n_joints} joints",
                    JOINT_STATE_INVALID.error_code,
                ))
            if not np.all(np.isfinite(values)):
                self._trip(JointStateInvalidError(f"joint state {field} contain NaN/inf", JOINT_STATE_INVALID.error_code))
        self._check_jump(state)
        if self._check_effort and state.efforts is None:
            self._trip(JointStateInvalidError(
                "joint state has no efforts, torque drift can't be watched. torque drivers must report joint effort",
                JOINT_STATE_INVALID.error_code,
            ))

    def _check_jump(self, state: JointState):
        """
        consecutive readings further apart than the joints can move in the time between them. two publishers on /joint_states
        (a stray joint_state_publisher sending zeros next to the driver) interleave like this, and every other reading would
        reset the tracking debounce, so the tracking check would never trip
        """
        previous = self._previous
        if previous is None or state.received_at > previous.received_at:
            self._previous = state
        if previous is None or state.received_at <= previous.received_at:
            return
        limits = self._parser.velocity_limits
        speed = np.where(limits > 0, limits * (1 + self._velocity_limit_margin), _UNSET_VELOCITY_LIMIT)
        allowed = speed * (state.received_at - previous.received_at) + _JUMP_SLACK_RAD
        jumped = self._names_where(np.abs(state.positions - previous.positions) > allowed)
        if jumped:
            worst = float(np.max(np.abs(state.positions - previous.positions)))
            self._trip(JointStateInvalidError(
                f"joints {jumped} jumped {worst:.3f}rad between two readings {state.received_at - previous.received_at:.3f}s "
                "apart, faster than they can move. is a second node publishing /joint_states (e.g. joint_state_publisher)?",
                JOINT_STATE_INVALID.error_code,
            ))

    def _check_measured_limits(self, state: JointState):
        margin = self._position_limit_margin_rad
        lower, upper = self._parser.position_limits
        lower, upper = lower - margin, upper + margin
        outside = self._names_where((state.positions < lower) | (state.positions > upper))
        if outside:
            self._trip(JointLimitError(
                f"measured positions of {outside} are outside the URDF limits. either the robot is past its limits or the URDF "
                "limits / encoder offsets are wrong, both would invalidate the calibration",
                JOINT_LIMIT.error_code,
            ))
        if state.velocities is not None:
            limits = self._parser.velocity_limits
            # URDFs leave unset limits at 0, those joints aren't checked. debounced: a driver's velocity is a finite difference,
            # and one late cycle (a controller_manager overrun) makes a single sample spike. a real overspeed lasts
            too_fast = self._names_where((limits > 0) & (np.abs(state.velocities) > limits * (1 + self._velocity_limit_margin)))
            self._debounce("velocity_limit", too_fast, lambda: JointLimitError(
                f"joints {too_fast} are moving faster than their URDF velocity limit for {self._max_consecutive_violations} "
                "readings in a row", JOINT_LIMIT.error_code))

    def _debounce(self, check: str, offending: list[str], make_error: Callable[[], CalibrationSafetyError]):
        if not offending:
            self._violations[check] = 0
            return
        self._violations[check] += 1
        if self._violations[check] >= self._max_consecutive_violations:
            self._trip(make_error())

    def _names_where(self, mask) -> list[str]:
        return [name for name, hit in zip(self._joint_names, mask) if hit]

    def _worst(self, values) -> str:
        i = int(np.argmax(values / self._effort_tolerance))
        return f"{values[i]:.2f}Nm on {self._joint_names[i]} (tolerance {self._effort_tolerance[i]:.2f}Nm)"

    def _raise_if_tripped(self):
        if self._tripped is not None:
            raise self._tripped

    def _trip(self, error: CalibrationSafetyError):
        try:
            self._safe_stop(error.message)
        except Exception as exc:
            failed = SafeStopFailedError(
                f"SAFE STOP FAILED, use the emergency stop. tried to stop because: {error.message}. the stop failed with: {exc}",
                SAFE_STOP_FAILED.error_code,
            )
            self._tripped = failed
            raise failed from exc
        except BaseException as interrupt:
            # ctrl-c or SIGTERM while the stop waited for the controller manager: whether the switch went through is unknown.
            # latch that, so the handlers further up report it instead of sending a second switch that would be refused
            self._tripped = SafeStopFailedError(
                f"SAFE STOP INTERRUPTED ({type(interrupt).__name__}) before the controller switch was confirmed. check the robot and "
                f"use the emergency stop if it is moving. tried to stop because: {error.message}",
                SAFE_STOP_FAILED.error_code,
            )
            raise
        error.stopped_safely = True
        error.message += ". the robot was stopped safely"
        error.args = (error.message,)
        self._tripped = error
        raise error


class ControllerSafeStop:
    """
    stops the arm with one strict ros2_control switch: deactivate the arm's controllers, activate the hold controller.

    everything slow happens in connect(), before anything moves: the rclpy node, the service client, and the check that the
    controller manager's switch service is there. A trip then only sends the request and waits for the answer.
    the node is our own and is spun by a private executor, so it never fights an executor the caller is already running.
    """

    def __init__(self, deactivate, activate=(), controller_manager: str = "/controller_manager", timeout_sec: float = 2.0):
        self.deactivate = list(deactivate)
        self.activate = list(activate)
        self._service_name = f"{controller_manager.rstrip('/')}/switch_controller"
        self._list_service_name = f"{controller_manager.rstrip('/')}/list_controllers"
        self._timeout_sec = timeout_sec
        self._node = None
        self._client = None
        self._list_client = None
        self._executor = None
        self._owns_context = False

    @property
    def connected(self) -> bool:
        return self._client is not None

    def connect(self) -> None:
        """
        creates the node and client and waits for the switch service. raises SafeStopUnconfiguredError if it isn't there,
        a stop that can't reach the controller manager is no stop at all.
        """
        if self.connected:
            return
        # imported here so the library still imports without a sourced ROS env
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from controller_manager_msgs.srv import ListControllers, SwitchController

        from .ros_context import init_rclpy

        # without rclpy's signal handlers, so ctrl-c / SIGTERM don't shut down the context this stop needs (ros_context.py)
        self._owns_context = init_rclpy()
        self._node = rclpy.create_node("calibration_safe_stop")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._client = self._node.create_client(SwitchController, self._service_name)
        self._list_client = self._node.create_client(ListControllers, self._list_service_name)
        for client, name in ((self._client, self._service_name), (self._list_client, self._list_service_name)):
            if not client.wait_for_service(timeout_sec=self._timeout_sec):
                self.close()
                raise SafeStopUnconfiguredError(
                    f"{name} not available after {self._timeout_sec}s, the robot couldn't be stopped through it",
                    SAFE_STOP_UNCONFIGURED.error_code,
                )
        self._request_type = SwitchController.Request
        self._list_request_type = ListControllers.Request

    def adopt_context(self, other: "ControllerSafeStop") -> None:
        """takes over the rclpy context other created, so closing other doesn't shut down the context this stop runs in"""
        if other._owns_context:
            other._owns_context, self._owns_context = False, True

    def close(self) -> None:
        """releases the node. after this the robot can't be stopped through this object, only call it once motion is over"""
        if self._node is not None:
            for client in (self._client, self._list_client):
                if client is not None:
                    self._node.destroy_client(client)
            self._executor.remove_node(self._node)
            self._executor.shutdown()
            self._node.destroy_node()
        self._node = self._client = self._list_client = self._executor = None
        if self._owns_context:
            import rclpy

            rclpy.shutdown()
            self._owns_context = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __call__(self, reason: str) -> None:
        """
        raises RuntimeError if the switch didn't happen, the monitor turns that into SafeStopFailedError.

        a refused STRICT switch isn't always a failed stop: a driver can switch the arm controller off by itself first (the
        UR driver's controller_stopper does when the robot program stops or a protective stop drops the reverse interface).
        So a refusal asks the controller manager what is active and retries only the part of the switch still needed.
        Nothing left to do means the controllers are already where the stop wanted them.
        """
        if not self.connected:
            raise RuntimeError("safe stop was never connected (or was closed), it can't reach the controller manager")
        # send first, log after, the log mustn't delay the stop
        future = self._client.call_async(self._switch_request(self.deactivate, self.activate))
        self._node.get_logger().error(f"safe stop: {reason}")
        if self._wait(future, self._service_name).ok:
            return
        active = self._active_controllers()
        still_on = [name for name in self.deactivate if name in active]
        not_on = [name for name in self.activate if name not in active]
        if not still_on and not not_on:
            self._node.get_logger().warning(f"safe stop: {self.deactivate} were already inactive, the driver switched them off")
            return
        if (still_on, not_on) != (self.deactivate, self.activate):
            retry = self._client.call_async(self._switch_request(still_on, not_on))
            if self._wait(retry, self._service_name).ok:
                return
        raise RuntimeError(
            f"controller manager refused to deactivate {self.deactivate} and activate {self.activate} "
            f"({still_on} still active, {not_on} not active)"
        )

    def _switch_request(self, deactivate, activate):
        request = self._request_type()
        request.deactivate_controllers = list(deactivate)
        request.activate_controllers = list(activate)
        request.strictness = self._request_type.STRICT
        request.activate_asap = True
        request.timeout.sec = int(self._timeout_sec)
        request.timeout.nanosec = int((self._timeout_sec % 1) * 1e9)
        return request

    def _wait(self, future, service_name: str):
        self._executor.spin_until_future_complete(future, timeout_sec=self._timeout_sec)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"{service_name} did not respond within {self._timeout_sec}s")
        return future.result()

    def _active_controllers(self) -> set[str]:
        answer = self._wait(self._list_client.call_async(self._list_request_type()), self._list_service_name)
        return {controller.name for controller in answer.controller if controller.state == "active"}


def build_safe_stop(hardware: HardwareControllerInterface, driver_type: DriverType, hold_controller: str | None = None,
                    timeout_sec: float = 2.0) -> ControllerSafeStop:
    """
    works out how this robot can be stopped safely, from the controllers the last driver type query found active.
    raises SafeStopUnconfiguredError when there is no safe way, so the run halts before anything moves.
    the returned stop isn't connected yet, call connect() (Orchestrator.setup() does) before moving the robot.
    """
    arm = hardware.joint_controllers()
    if not arm:
        raise SafeStopUnconfiguredError(
            "no active controller claims the arm's joints, so there is nothing to switch off to stop the robot. query the "
            "driver type first and check the arm controller is active",
            SAFE_STOP_UNCONFIGURED.error_code,
        )
    if hold_controller is not None:
        if hold_controller in arm:
            raise SafeStopUnconfiguredError(
                f"hold controller '{hold_controller}' is the controller the babble runs on, a stop would switch it off. "
                "name a separate, inactive controller that holds the arm in place",
                SAFE_STOP_UNCONFIGURED.error_code,
            )
        return ControllerSafeStop(arm, [hold_controller], hardware.controller_manager, timeout_sec)

    if driver_type == DriverType.TORQUE:
        raise SafeStopUnconfiguredError(
            f"the arm runs on effort controllers {list(arm)}. switching them off removes the holding torque and the arm can "
            "drop, so a torque robot needs hold_controller: an inactive controller that holds position on its own "
            "(e.g. a joint_trajectory_controller with effort PID gains)",
            SAFE_STOP_UNCONFIGURED.error_code,
        )
    not_position = sorted({i.rpartition("/")[2] for interfaces in arm.values() for i in interfaces} - {"position"})
    if not_position:
        raise SafeStopUnconfiguredError(
            f"the arm controllers {list(arm)} command {not_position}. switching them off may leave the last command applied, "
            "so name a hold_controller to switch to instead",
            SAFE_STOP_UNCONFIGURED.error_code,
        )
    # position controllers: the hardware keeps holding the last commanded position once they're off
    return ControllerSafeStop(arm, [], hardware.controller_manager, timeout_sec)


class JointStateListener:
    """keeps the latest /joint_states reading. the node has to be spun for it to update"""

    def __init__(self, node, joint_names, topic: str = "/joint_states"):
        from sensor_msgs.msg import JointState as JointStateMsg

        self._node = node
        self._joint_names = list(joint_names)
        self._latest = None
        self._last_stamp = None
        self._error: CalibrationSafetyError | None = None
        self._subscription = node.create_subscription(JointStateMsg, topic, self._on_msg, 10)

    def _on_msg(self, msg):
        # a reading whose header stamp doesn't advance (latched, relayed or republished) is not new feedback: dropping it lets
        # the age check see that the robot isn't being reported. drivers that leave the stamp at zero are taken as they come
        header = getattr(msg, "header", None)
        stamp = header.stamp.sec + header.stamp.nanosec * 1e-9 if header is not None else 0.0
        if stamp > 0.0:
            if self._last_stamp is not None and stamp <= self._last_stamp:
                return
            self._last_stamp = stamp
        try:
            self._latest = joint_state_from_msg(msg, self._joint_names, time.monotonic())
        except CalibrationSafetyError as error:
            self._error = error

    def close(self) -> None:
        if self._subscription is not None:
            self._node.destroy_subscription(self._subscription)
            self._subscription = None

    def latest(self) -> JointState:
        if self._error is not None:
            raise self._error
        if self._latest is None:
            raise JointStateInvalidError("no /joint_states received yet", JOINT_STATE_INVALID.error_code)
        return self._latest
