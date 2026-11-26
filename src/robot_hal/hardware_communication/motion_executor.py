"""
moves the robot from where it is to a joint space goal, with the safety monitor watching every joint state on the way.

we don't stream setpoints to the drivers. ros2_control's joint_trajectory_controller already runs the real time loop at the drive's
rate (mvp.md §9 keeps our own control loop out of scope), so each move is one whole trajectory sent as a FollowJointTrajectory
action goal ("scaled_joint_trajectory_controller" on UR). this file:
    1. plans a quintic joint space move at a fraction of the URDF velocity limits (plan_move, pure, no ros2)
    2. checks every planned point against the URDF limits, through formulas.command, and an optional waypoint_check
       (e.g. Babble's workspace box: a joint space path between two poses inside the box can leave it in between)
    3. sends it, then feeds every new joint state to monitor.check with the controller's own reference as the setpoint
    4. cancels the goal the moment the monitor trips, or stops through the monitor when the controller reports a failure
the setpoint comes from the controller's state topic, never from wall time: UR speed scaling slows the trajectory down, and a
time indexed setpoint would run ahead of the real one and trip the tracking check.

the transport (JointTrajectoryTransport) is the only ros2 part, MotionExecutor works against the small interface it implements,
so the motion logic is tested without ros2. the transport's node has to be spun in the background (BackgroundExecutor).

on torque drivers the trajectory controller computes the efforts itself (PID on the trajectory), so the commands formulas.command
builds are only used to check the plan against the effort limits, they aren't passed to monitor.check as the delivered command.
"""
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from ..errors.motor_controller_errors import ControllerDriversUnreachableError, DRIVER_UNREACHABLE_ERROR
from ..errors.safety_errors import (
    CalibrationSafetyError, JointLimitError, JointStateInvalidError, JOINT_LIMIT, JOINT_STATE_INVALID,
)
from .safety_monitor import JointState, SafetyMonitor

# peak velocity and acceleration of a rest to rest quintic, as multiples of distance/T and distance/T^2
_QUINTIC_PEAK_VELOCITY = 1.875
_QUINTIC_PEAK_ACCELERATION = 10.0 / math.sqrt(3.0)


@dataclass(frozen=True)
class TrajectoryPoint:
    """one point of a planned move, in joint space (URDF joint order)"""
    time_from_start: float
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray


@dataclass(frozen=True)
class ControllerReference:
    """where the trajectory controller is steering the joints right now, in URDF joint order"""
    positions: np.ndarray
    accelerations: np.ndarray | None = None


@dataclass(frozen=True)
class MoveResult:
    """how the move ended. a move that didn't finish raises instead, so this always means the goal was reached"""
    duration_sec: float
    final_state: JointState
    max_goal_error_rad: float


class TrajectoryTransport(Protocol):
    """what MotionExecutor needs from the trajectory controller. JointTrajectoryTransport is the ros2 implementation"""

    def send(self, points: list[TrajectoryPoint]) -> None:
        """sends the trajectory and waits for the controller to accept it, raises if it is rejected"""

    def reference(self) -> ControllerReference | None:
        """the controller's latest reference for the running goal, None until one has arrived"""

    def done(self) -> bool:
        """the controller has finished the goal (succeeded, failed or was cancelled)"""

    def result(self) -> tuple[bool, str]:
        """(succeeded, message) once done()"""

    def cancel(self) -> None:
        """asks the controller to stop the goal. must not raise, it runs while a safety error is already on its way up"""


def plan_move(start, goal, velocity_limits, velocity_fraction: float = 0.1, max_acceleration_rad_s2: float = 0.5,
              default_velocity_rad_s: float = 0.2, min_duration_sec: float = 0.5,
              point_spacing_sec: float = 0.1) -> list[TrajectoryPoint]:
    """
    a rest to rest quintic in joint space, every joint starting and arriving together.
    the duration is the shortest that keeps every joint under velocity_fraction of its URDF velocity limit (default_velocity_rad_s
    where the URDF leaves it unset) and under max_acceleration_rad_s2. points are spaced point_spacing_sec apart and the last one
    is exactly the goal. the start isn't included, the trajectory controller starts from where the joints are.
    """
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if start.shape != goal.shape:
        raise ValueError(f"start and goal have different shapes, {start.shape} vs {goal.shape}")
    if not 0.0 < velocity_fraction <= 1.0:
        raise ValueError(f"velocity_fraction must be in (0, 1], got {velocity_fraction}")
    limits = np.asarray(velocity_limits, dtype=float)
    max_velocity = np.where(limits > 0.0, velocity_fraction * limits, default_velocity_rad_s)

    distance = np.abs(goal - start)
    duration = max(
        min_duration_sec,
        float(np.max(_QUINTIC_PEAK_VELOCITY * distance / max_velocity)),
        float(np.max(np.sqrt(_QUINTIC_PEAK_ACCELERATION * distance / max_acceleration_rad_s2))),
    )
    steps = max(1, math.ceil(duration / point_spacing_sec))
    delta = goal - start
    points = []
    for k in range(1, steps + 1):
        t = duration * k / steps
        s = t / duration
        # s(τ) = 10τ³ − 15τ⁴ + 6τ⁵ and its derivatives
        position = 10 * s**3 - 15 * s**4 + 6 * s**5
        velocity = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
        acceleration = (60 * s - 180 * s**2 + 120 * s**3) / duration**2
        points.append(TrajectoryPoint(
            t,
            goal.copy() if k == steps else start + position * delta,
            np.zeros_like(delta) if k == steps else velocity * delta,
            np.zeros_like(delta) if k == steps else acceleration * delta,
        ))
    return points


class _Path:
    """the straight joint space segment a rest to rest move follows, and whether a reference stays on it, going forward"""

    def __init__(self, start, goal, tolerance_rad: float):
        self.start = np.asarray(start, dtype=float)
        self.delta = np.asarray(goal, dtype=float) - self.start
        self.length2 = float(self.delta @ self.delta)
        self.tolerance = tolerance_rad
        self.progress = 0.0

    def off_path(self, reference) -> str | None:
        """None while the reference is on the segment and not going back, otherwise what is wrong"""
        reference = np.asarray(reference, dtype=float)
        if self.length2 < 1e-12:
            off = float(np.max(np.abs(reference - self.start)))
            return None if off <= self.tolerance else f"moved {off:.4f}rad on a move that goes nowhere"
        s = float((reference - self.start) @ self.delta) / self.length2
        nearest = self.start + np.clip(s, 0.0, 1.0) * self.delta
        off = float(np.max(np.abs(reference - nearest)))
        if off > self.tolerance:
            return f"is {off:.4f}rad off the planned path (tolerance {self.tolerance}rad)"
        # in path units, so the slack is the tolerance relative to the longest joint motion
        slack = self.tolerance / max(float(np.max(np.abs(self.delta))), 1e-9)
        if s < self.progress - slack:
            return f"went back along the planned path (from {self.progress:.3f} to {s:.3f})"
        self.progress = max(self.progress, s)
        return None


class MotionExecutor:
    def __init__(self, parser, monitor: SafetyMonitor, transport: TrajectoryTransport, joint_states,
                 waypoint_check: Callable[[np.ndarray], None] | None = None, velocity_fraction: float = 0.1,
                 max_acceleration_rad_s2: float = 0.5, default_velocity_rad_s: float = 0.2, point_spacing_sec: float = 0.1,
                 goal_tolerance_rad: float = 0.005, reference_timeout_sec: float = 1.0, finish_slack: float = 10.0,
                 finish_margin_sec: float = 5.0, poll_period_sec: float = 0.005, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, alive: Callable[[], bool] | None = None,
                 speed_cap_factor: float = 1.5, speed_cap_floor_rad_s: float = 0.05, path_tolerance_rad: float = 0.01,
                 start_tolerance_rad: float = 0.01, start_velocity_rad_s: float = 0.05, start_settle_sec: float = 2.0,
                 goal_settle_sec: float = 1.0, stall_timeout_sec: float = 3.0, stall_tolerance_rad: float = 1e-6,
                 robot_status: Callable[[], str | None] | None = None):
        """
        parser: URDFParser with formulas generated. its formulas check the plan, its limits bound it.
        monitor: the run's SafetyMonitor, every new joint state during a move goes through monitor.check.
        transport: the trajectory controller, e.g. JointTrajectoryTransport.
        joint_states: anything with latest() -> JointState, e.g. safety_monitor.JointStateListener.
        waypoint_check: called with every planned position, raises to refuse the move before anything is sent (workspace box,
            collision check). the robot doesn't move if any point fails.
        velocity_fraction, max_acceleration_rad_s2, default_velocity_rad_s, point_spacing_sec: see plan_move.
        goal_tolerance_rad: how close every joint has to end up to the goal.
        reference_timeout_sec: how long the controller may take to report a reference after accepting the goal.
        finish_slack / finish_margin_sec: the move has to finish within finish_slack * planned duration + finish_margin_sec. speed
            scaling (UR's speed slider) stretches a move, the default allows a slider down to about 0.1. a stalled controller is
            caught sooner, by the stall check.
        stall_timeout_sec / stall_tolerance_rad: a reference that moves less than stall_tolerance_rad on every joint for
            stall_timeout_sec before the goal is done is a stall (a paused program, speed scaling at zero): the robot is stopped.
        alive: whether the joint states can still arrive, e.g. BackgroundExecutor.running. checked on every poll.
        speed_cap_factor / speed_cap_floor_rad_s: a joint moving faster than max(factor x its planned peak speed, peak + floor)
            trips the monitor. the URDF limit alone allows many times the planned speed.
        path_tolerance_rad: how far the controller's reference may leave the planned joint space segment. every move is a
            straight segment, so a reference off it is another trajectory than the one the guard checked.
        start_tolerance_rad: how far the measured start may be from the controller's own feedback before the goal is sent.
        start_velocity_rad_s / start_settle_sec: the robot must be under this speed before a goal is sent, waiting up to this
            long for it to settle.
        goal_settle_sec: once the controller reports the goal done, how long the joints may take to come within
            goal_tolerance_rad of it (the arm lags the reference at the end of a move), watched by the monitor meanwhile.
        robot_status: returns why the robot can't follow a move (protective stop, paused program, powered off), or None while it
            can, e.g. robot_status.URRobotStatus.problem. checked before a goal is sent and on every poll, so a stop the robot
            made itself is seen at once instead of through the stall check or the finish deadline.
        """
        self._parser = parser
        self._monitor = monitor
        self._transport = transport
        self._joint_states = joint_states
        self._waypoint_check = waypoint_check
        self._plan_options = dict(velocity_fraction=velocity_fraction, max_acceleration_rad_s2=max_acceleration_rad_s2,
                                  default_velocity_rad_s=default_velocity_rad_s, point_spacing_sec=point_spacing_sec)
        self._goal_tolerance_rad = goal_tolerance_rad
        self._reference_timeout_sec = reference_timeout_sec
        self._finish_slack = finish_slack
        self._finish_margin_sec = finish_margin_sec
        self._poll_period_sec = poll_period_sec
        self._clock = clock
        self._sleep = sleep
        self._alive = alive
        self._speed_cap_factor = speed_cap_factor
        self._speed_cap_floor = speed_cap_floor_rad_s
        self._path_tolerance_rad = path_tolerance_rad
        self._start_tolerance_rad = start_tolerance_rad
        self._start_velocity_rad_s = start_velocity_rad_s
        self._start_settle_sec = start_settle_sec
        self._goal_settle_sec = goal_settle_sec
        self._stall_timeout_sec = stall_timeout_sec
        self._stall_tolerance_rad = stall_tolerance_rad
        self._robot_status = robot_status

    def plan(self, goal, start=None) -> list[TrajectoryPoint]:
        """plans and checks a move, without moving. start defaults to the measured joint positions"""
        if start is None:
            start = self._joint_states.latest().positions
        goal = np.asarray(goal, dtype=float)
        self._check_position_limits(goal, "goal")
        points = plan_move(start, goal, self._parser.velocity_limits, **self._plan_options)
        formulas = self._parser.formulas
        for point in points:
            # position limits are checked here for every driver type, the torque command formula only checks efforts
            self._check_position_limits(point.positions, f"planned point at t={point.time_from_start:.2f}s")
            formulas.command(point.positions, point.velocities, point.accelerations)
            if self._waypoint_check is not None:
                self._waypoint_check(point.positions)
        return points

    def move_to(self, goal) -> MoveResult:
        """
        moves to goal and returns once the joints are there. raises:
            the monitor's latched error, before anything is planned, if it has already tripped
            JointLimitError (or whatever waypoint_check raises) before anything moves, if the plan is refused
            ControllerDriversUnreachableError if the controller rejects the goal, before anything moves
            any CalibrationSafetyError from the monitor (stale, NaN, moving or misplaced start; then during the move), after
                stopping the robot and cancelling the goal
            CalibrationAbortedError (through monitor.stop) if the controller fails the goal, stalls, stops off the goal, leaves
                the planned path, feedback stops arriving or robot_status reports a problem
        anything else raised during the move (ctrl-c, SIGTERM, a bug, a goal accepted too late) stops the robot through the
        monitor first, then cancels the goal, then re-raises.
        """
        tripped = self._monitor.tripped
        if tripped is not None:
            raise tripped
        start_state = self._ready_state()
        start = np.asarray(start_state.positions, dtype=float)
        goal = np.asarray(goal, dtype=float)
        points = self.plan(goal, start)
        planned = points[-1].time_from_start
        peak = np.max(np.abs([p.velocities for p in points]), axis=0)
        velocity_cap = np.maximum(self._speed_cap_factor * peak, peak + self._speed_cap_floor)
        deadline = self._clock() + self._finish_slack * planned + self._finish_margin_sec

        started = self._clock()
        last_state = start_state
        path = _Path(start, goal, self._path_tolerance_rad)
        try:
            self._transport.send(points)
            # the reference timeout counts from acceptance: send() may have waited for it, and the controller can't report
            # a reference for a goal it hasn't accepted yet
            accepted = self._clock()
            # the stall check: the last reference that moved, and when
            moved_reference, moved_at = None, accepted
            while True:
                self._check_robot_status()
                state = self._fresh_state(last_state)
                if state is not last_state:
                    last_state = state
                    reference = self._transport.reference()
                    if reference is None and self._clock() - accepted > self._reference_timeout_sec:
                        self._monitor.stop(
                            f"the trajectory controller accepted the goal but reported no reference for "
                            f"{self._reference_timeout_sec}s (controller_state silent or stale), the robot can't be tracked"
                        )
                    if reference is not None:
                        if (moved_reference is None
                                or np.max(np.abs(reference.positions - moved_reference)) > self._stall_tolerance_rad):
                            moved_reference, moved_at = np.array(reference.positions, dtype=float), self._clock()
                        elif self._clock() - moved_at > self._stall_timeout_sec:
                            self._monitor.stop(
                                f"the trajectory controller's reference hasn't moved for {self._stall_timeout_sec}s and the "
                                "goal isn't done: the robot program is paused, the speed slider is at zero or the controller "
                                "stalled"
                            )
                        off = path.off_path(reference.positions)
                        if off is not None:
                            self._monitor.stop(f"the trajectory controller's reference {off}. it is running another "
                                               "trajectory than the one that was checked")
                    self._monitor.check(
                        state,
                        setpoint=None if reference is None else reference.positions,
                        acceleration=None if reference is None else reference.accelerations,
                        velocity_cap=velocity_cap,
                    )
                else:
                    # no new reading: the monitor only sees fresh ones, so staleness has to be caught here, every poll
                    age = self._clock() - last_state.received_at
                    if age > self._monitor.max_state_age_sec:
                        self._monitor.stop(f"no new joint state for {age:.3f}s during the move (limit "
                                           f"{self._monitor.max_state_age_sec}s), the robot can't be watched")
                if self._alive is not None and not self._alive():
                    self._monitor.stop("the ROS executor thread stopped, joint states and the controller state no longer "
                                       "arrive")
                if self._transport.done():
                    break
                if self._clock() > deadline:
                    self._monitor.stop(
                        f"the move didn't finish within {deadline - started:.1f}s (planned {planned:.1f}s), the trajectory "
                        "controller stalled or the speed slider is near zero"
                    )
                self._sleep(self._poll_period_sec)
        except CalibrationSafetyError:
            # a trip has already stopped the robot, cancel so the controller doesn't resume the goal
            self._transport.cancel()
            raise
        except BaseException as error:
            # ctrl-c, SIGTERM (ros_context.Shutdown), a goal accepted after the wait gave up, a bug: the goal may be running
            # and nothing watches it any more. stop the robot first, then cancel. a rejected goal never ran, nothing to stop
            try:
                if getattr(error, "goal_may_run", True):
                    self._stop_for(error)
            finally:
                # even when the stop failed: cancelling is then the last way left to end the motion
                self._transport.cancel()
            raise

        succeeded, message = self._transport.result()
        if not succeeded:
            self._monitor.stop(f"the trajectory controller didn't complete the move: {message}")
        final = self._settle_at(goal)
        error = np.abs(final.positions - goal)
        if np.max(error) > self._goal_tolerance_rad:
            worst = self._parser.joint_names[int(np.argmax(error))]
            self._monitor.stop(
                f"the move finished {np.max(error):.4f}rad off the goal on {worst} (tolerance {self._goal_tolerance_rad}rad)"
            )
        return MoveResult(self._clock() - started, final, float(np.max(error)))

    def _settle_at(self, goal) -> JointState:
        """the joint state once every joint is within goal_tolerance_rad of goal, or the last one after goal_settle_sec. every
        new reading goes through the monitor, and readings stopping is a stop, as during the move"""
        final = self._fresh_state(None)
        self._monitor.check(final, setpoint=goal)
        deadline = self._clock() + self._goal_settle_sec
        while np.max(np.abs(final.positions - goal)) > self._goal_tolerance_rad and self._clock() < deadline:
            self._sleep(self._poll_period_sec)
            state = self._fresh_state(final)
            if state is not final:
                self._monitor.check(state, setpoint=goal)
                final = state
            elif self._clock() - final.received_at > self._monitor.max_state_age_sec:
                self._monitor.stop(f"no new joint state for {self._clock() - final.received_at:.3f}s at the end of the move "
                                   f"(limit {self._monitor.max_state_age_sec}s), the robot can't be watched")
        return final

    def _ready_state(self) -> JointState:
        """
        the start of a move, checked before anything is sent: fresh, finite and inside the limits (monitor.check), the robot
        settled, and where the controller itself thinks it is. the plan and the guard only cover the path from this reading,
        the controller interpolates from where the arm really is to the first point, so a wrong start is an unchecked segment
        """
        state = self._fresh_state(None)
        self._monitor.check(state)
        self._check_robot_status()
        if state.velocities is None:
            # without velocities the settle wait, the planned-speed cap and the URDF velocity check would all be off
            raise JointStateInvalidError("/joint_states has no velocities, a move can't be watched for speed. use a driver that "
                                         "publishes them", JOINT_STATE_INVALID.error_code)
        deadline = self._clock() + self._start_settle_sec
        while state.velocities is not None and np.any(np.abs(state.velocities) > self._start_velocity_rad_s):
            if self._clock() > deadline:
                self._monitor.stop(f"the robot is still moving (up to {np.max(np.abs(state.velocities)):.3f}rad/s) "
                                   f"{self._start_settle_sec}s after it should have stopped, a new move can't start")
            self._sleep(self._poll_period_sec)
            fresh = self._fresh_state(state)
            if fresh is not state:
                self._monitor.check(fresh)
            state = fresh
        feedback = getattr(self._transport, "feedback", None)
        if not callable(feedback):
            # a transport that can't report the controller's view (a simulated one) skips the comparison. one that can but
            # has nothing recent can't confirm where the arm is, and the controller would start from there unchecked (§2.8)
            return state
        controller = feedback()
        if controller is None:
            self._monitor.stop(
                "the trajectory controller hasn't reported where the arm is in the last second (controller_state silent or "
                "without measured positions), so the start of the move can't be checked against it"
            )
        else:
            gap = np.abs(np.asarray(state.positions, dtype=float) - np.asarray(controller, dtype=float))
            if np.max(gap) > self._start_tolerance_rad:
                worst = self._parser.joint_names[int(np.argmax(gap))]
                self._monitor.stop(
                    f"/joint_states and the trajectory controller disagree by {np.max(gap):.4f}rad on {worst} before the move "
                    f"(tolerance {self._start_tolerance_rad}rad). a second /joint_states publisher or a stale reading, the "
                    "planned path wouldn't start where the arm is"
                )
        return state

    def _check_robot_status(self) -> None:
        if self._robot_status is None:
            return
        problem = self._robot_status()
        if problem is not None:
            self._monitor.stop(f"the robot reports {problem}, it can't follow a move")

    def _stop_for(self, error: BaseException) -> None:
        """the safe stop for an exception that isn't a monitor trip. a failed stop (SafeStopFailedError) wins over the error"""
        try:
            self._monitor.stop(f"the move was interrupted ({type(error).__name__}: {error})")
        except CalibrationSafetyError as stopped:
            if not getattr(stopped, "stopped_safely", False) and self._monitor.tripped is stopped:
                raise stopped from error

    def _fresh_state(self, previous: JointState | None) -> JointState:
        """
        the latest joint state. the same reading must not be checked twice, it would count twice toward a debounced trip.
        while moving (previous given), a listener that can't produce a state stops the robot through the monitor, raising its
        error directly would leave the arm moving unwatched.
        """
        try:
            state = self._joint_states.latest()
        except CalibrationSafetyError as error:
            if previous is None:
                raise
            self._monitor.stop(f"lost joint state feedback during the move: {error.message}")
        if previous is not None and state.received_at == previous.received_at:
            return previous
        return state

    def _check_position_limits(self, q, what: str):
        lower, upper = self._parser.position_limits
        outside = [name for name, value, lo, hi in zip(self._parser.joint_names, q, lower, upper) if not lo <= value <= hi]
        if outside:
            raise JointLimitError(f"{what} is outside the URDF limits for {outside}", JOINT_LIMIT.error_code)


class JointTrajectoryTransport:
    """
    the ros2 side: a FollowJointTrajectory action client on <controller>/follow_joint_trajectory and a subscription to
    <controller>/controller_state for the reference. the node must be spun in the background (BackgroundExecutor), every wait
    here polls a future instead of spinning.
    """

    def __init__(self, node, controller: str, joint_names, timeout_sec: float = 5.0, goal_time_tolerance_sec: float = 1.0,
                 max_state_age_sec: float = 0.5):
        """
        controller: the arm's trajectory controller, e.g. "scaled_joint_trajectory_controller" on UR.
        joint_names: URDF joint order, the reference is reordered into it.
        timeout_sec: how long the action server and goal acceptance may take.
        goal_time_tolerance_sec: how late the controller may reach the goal before it fails the goal itself.
        max_state_age_sec: a controller_state older than this isn't returned as the reference, so a frozen topic can't keep the
            path and tracking checks comparing against where the controller was.
        """
        self._node = node
        self._controller = controller.strip("/")
        self._joint_names = list(joint_names)
        self._timeout_sec = timeout_sec
        self._goal_time_tolerance_sec = goal_time_tolerance_sec
        self._client = None
        self._subscription = None
        self._lock = threading.Lock()
        self._max_state_age_sec = max_state_age_sec
        self._reference: tuple[float, ControllerReference] | None = None    # (received at, reference)
        self._goal_handle = None
        self._result_future = None
        self._feedback: tuple[float, np.ndarray] | None = None     # (received at, positions), from controller_state

    def connect(self) -> None:
        """creates the client and subscription and waits for the action server. raises ControllerDriversUnreachableError"""
        if self._client is not None:
            return
        # imported here so the library still imports without a sourced ROS env
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory
        from control_msgs.msg import JointTrajectoryControllerState

        self._action_type = FollowJointTrajectory
        action = f"/{self._controller}/follow_joint_trajectory"
        self._client = ActionClient(self._node, FollowJointTrajectory, action)
        self._subscription = self._node.create_subscription(
            JointTrajectoryControllerState, f"/{self._controller}/controller_state", self._on_state, 10
        )
        if not self._client.wait_for_server(timeout_sec=self._timeout_sec):
            self.close()
            raise ControllerDriversUnreachableError(
                f"{action} not available after {self._timeout_sec}s. is '{self._controller}' loaded and active?",
                DRIVER_UNREACHABLE_ERROR.error_code,
            )

    def close(self) -> None:
        if self._subscription is not None:
            self._node.destroy_subscription(self._subscription)
        if self._client is not None:
            self._client.destroy()
        self._client = self._subscription = None

    def send(self, points: list[TrajectoryPoint]) -> None:
        if self._client is None:
            self.connect()
        goal = self._action_type.Goal()
        goal.trajectory = self._trajectory_msg(points)
        goal.goal_time_tolerance = _duration_msg(self._goal_time_tolerance_sec)
        with self._lock:
            # a reference left over from the previous goal would be read as this goal's setpoint
            self._reference = None
        self._goal_handle = None
        self._result_future = None
        future = self._client.send_goal_async(goal)
        try:
            self._goal_handle = self._wait(future, "accept the goal")
        except BaseException:
            # it may still be accepted after the wait gave up, and would then run with nothing watching: cancel it on arrival
            future.add_done_callback(_cancel_if_accepted)
            raise
        if not self._goal_handle.accepted:
            error = ControllerDriversUnreachableError(
                f"'{self._controller}' rejected the trajectory. the controller may be inactive or the joints don't match",
                DRIVER_UNREACHABLE_ERROR.error_code,
            )
            error.goal_may_run = False
            raise error
        self._result_future = self._goal_handle.get_result_async()

    def reference(self) -> ControllerReference | None:
        """the controller's reference for the running goal, if its controller_state arrived in the last max_state_age_sec"""
        with self._lock:
            if self._reference is None or time.monotonic() - self._reference[0] > self._max_state_age_sec:
                return None
            return self._reference[1]

    def feedback(self) -> np.ndarray | None:
        """where the controller measures the joints (URDF order), if its controller_state arrived in the last second"""
        with self._lock:
            if self._feedback is None or time.monotonic() - self._feedback[0] > 1.0:
                return None
            return self._feedback[1].copy()

    def done(self) -> bool:
        return self._result_future is not None and self._result_future.done()

    def result(self) -> tuple[bool, str]:
        response = self._result_future.result()
        result = response.result
        succeeded = result.error_code == self._action_type.Result.SUCCESSFUL
        message = f"error_code {result.error_code}" + (f": {result.error_string}" if result.error_string else "")
        return succeeded, message

    def cancel(self) -> None:
        try:
            if self._goal_handle is not None and not self.done():
                self._goal_handle.cancel_goal_async()
        except Exception as exc:
            # the safe stop already switched controllers, a failed cancel only leaves a dead goal behind
            self._node.get_logger().warning(f"cancelling the trajectory goal failed: {exc}")

    def _on_state(self, msg):
        # jazzy and later name it reference, humble names it desired
        point = msg.reference if hasattr(msg, "reference") else msg.desired
        names = list(msg.joint_names)
        try:
            order = [names.index(joint) for joint in self._joint_names]
        except ValueError:
            return
        positions = np.asarray(point.positions, dtype=float)
        if positions.shape != (len(names),):
            return
        accelerations = np.asarray(point.accelerations, dtype=float)
        reference = ControllerReference(
            positions[order], accelerations[order] if accelerations.shape == (len(names),) else None
        )
        # the measured positions: feedback on jazzy and later, actual on humble
        measured_point = msg.feedback if hasattr(msg, "feedback") else getattr(msg, "actual", None)
        measured = np.asarray(measured_point.positions, dtype=float) if measured_point is not None else np.empty(0)
        with self._lock:
            self._reference = (time.monotonic(), reference)
            if measured.shape == (len(names),):
                self._feedback = (time.monotonic(), measured[order])

    def _trajectory_msg(self, points: list[TrajectoryPoint]):
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        trajectory = JointTrajectory()
        trajectory.joint_names = self._joint_names
        for point in points:
            msg = JointTrajectoryPoint()
            msg.positions = point.positions.tolist()
            msg.velocities = point.velocities.tolist()
            msg.accelerations = point.accelerations.tolist()
            msg.time_from_start = _duration_msg(point.time_from_start)
            trajectory.points.append(msg)
        return trajectory

    def _wait(self, future, what: str):
        deadline = time.monotonic() + self._timeout_sec
        while not future.done():
            if time.monotonic() > deadline:
                raise ControllerDriversUnreachableError(
                    f"'{self._controller}' didn't {what} within {self._timeout_sec}s. is the node spun (BackgroundExecutor)?",
                    DRIVER_UNREACHABLE_ERROR.error_code,
                )
            time.sleep(0.005)
        return future.result()


def _cancel_if_accepted(future) -> None:
    try:
        handle = future.result()
        if handle is not None and handle.accepted:
            handle.cancel_goal_async()
    except Exception:
        pass


def _duration_msg(seconds: float):
    from builtin_interfaces.msg import Duration

    sec, nanosec = divmod(int(round(seconds * 1e9)), 1_000_000_000)
    return Duration(sec=sec, nanosec=nanosec)
