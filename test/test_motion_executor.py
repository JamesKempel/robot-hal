import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from graph_calibration import URDFParser
from robot_hal.errors import (
    CalibrationAbortedError,
    ControllerDriversUnreachableError,
    JointLimitError,
    TrackingError,
)
from robot_hal.hardware_communication.hardware_controller_interface import DriverType
from robot_hal.hardware_communication.motion_executor import (
    ControllerReference,
    JointTrajectoryTransport,
    MotionExecutor,
    _QUINTIC_PEAK_ACCELERATION,
    plan_move,
)
from robot_hal.hardware_communication.safety_monitor import JointState, SafetyMonitor

URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ur20.urdf")
UR_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
START = np.array([0.3, -1.2, 1.1, -0.8, 1.4, 0.5])
GOAL = START + np.array([0.2, -0.1, 0.15, 0.0, -0.2, 0.3])
DT = 0.01


class RecordingStop:
    def __init__(self):
        self.reasons = []

    def __call__(self, reason):
        self.reasons.append(reason)


class Sim:
    """
    a fake clock, a fake trajectory controller and fake joint feedback in one. every sleep advances time by DT and produces a new
    joint state. by default the robot tracks the trajectory perfectly, the flags make it misbehave.
    """

    def __init__(self, start=START):
        self.now = 0.0
        self.start = np.asarray(start, dtype=float)
        self.points = None
        self.sent_at = None
        self.cancelled = False
        self.succeeds = True
        self.finishes = True
        self.reports_reference = True
        self.robot_stalls = False
        self.feedback_lost_after = None
        self.final_offset = np.zeros(6)
        self.received_at = 0.0

    # clock
    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += DT
        self.received_at = self.now

    def _planned(self, t):
        if self.points is None or t <= 0:
            return self.start.copy()
        for point in self.points:
            if point.time_from_start >= t:
                return point.positions.copy()
        return self.points[-1].positions.copy()

    def _elapsed(self):
        return self.now - self.sent_at if self.sent_at is not None else 0.0

    # joint states
    def latest(self):
        if self.feedback_lost_after is not None and self._elapsed() > self.feedback_lost_after:
            from robot_hal.errors import JointStateInvalidError
            raise JointStateInvalidError("no /joint_states received yet", 304)
        q = self.start.copy() if self.robot_stalls else self._planned(self._elapsed())
        if self.done():
            q = q + self.final_offset
        return JointState(self.received_at, q, np.zeros(6))

    # transport
    def send(self, points):
        self.points = points
        self.sent_at = self.now

    def reference(self):
        if not self.reports_reference or self.sent_at is None:
            return None
        return ControllerReference(self._planned(self._elapsed()))

    def done(self):
        return self.finishes and self.points is not None and self._elapsed() >= self.points[-1].time_from_start

    def result(self):
        return (True, "") if self.succeeds else (False, "error_code -4: PATH_TOLERANCE_VIOLATED")

    def cancel(self):
        self.cancelled = True


def make_executor(sim=None, waypoint_check=None, monitor=None, **kwargs):
    parser = URDFParser(URDF_PATH)
    parser.generate_initial_formulas(DriverType.POS_VELO)
    sim = sim or Sim()
    stop = RecordingStop()
    monitor = monitor or SafetyMonitor(parser, stop, clock=sim.clock)
    executor = MotionExecutor(parser, monitor, sim, sim, waypoint_check=waypoint_check, velocity_fraction=0.5,
                              clock=sim.clock, sleep=sim.sleep, **kwargs)
    return executor, sim, stop, parser


class TestPlanMove:

    def test_ends_exactly_at_goal_at_rest(self):
        points = plan_move(START, GOAL, np.full(6, 2.0))
        np.testing.assert_array_equal(points[-1].positions, GOAL)
        assert not points[-1].velocities.any() and not points[-1].accelerations.any()
        assert points[0].time_from_start > 0, "the start isn't sent, the controller starts from the measured position"

    def test_respects_velocity_and_acceleration(self):
        limits = np.array([2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
        points = plan_move(START, GOAL, limits, velocity_fraction=0.1, max_acceleration_rad_s2=0.5, point_spacing_sec=0.01)
        velocities = np.array([p.velocities for p in points])
        accelerations = np.array([p.accelerations for p in points])
        assert np.all(np.abs(velocities) <= 0.1 * limits + 1e-9)
        assert np.all(np.abs(accelerations) <= 0.5 + 1e-9)

    def test_velocity_matches_positions(self):
        points = plan_move(START, GOAL, np.full(6, 2.0), point_spacing_sec=0.001)
        positions = np.array([p.positions for p in points])
        times = np.array([p.time_from_start for p in points])
        numeric = np.gradient(positions, times, axis=0)
        np.testing.assert_allclose(numeric[5:-5], np.array([p.velocities for p in points])[5:-5], atol=1e-3)

    def test_acceleration_bound_sets_duration_for_short_moves(self):
        tiny = START + 1e-3
        duration = plan_move(START, tiny, np.full(6, 2.0), max_acceleration_rad_s2=0.5, min_duration_sec=0.0)[-1].time_from_start
        assert duration == pytest.approx(np.sqrt(_QUINTIC_PEAK_ACCELERATION * 1e-3 / 0.5))

    def test_unset_limits_use_default(self):
        points = plan_move(START, GOAL, np.zeros(6), default_velocity_rad_s=0.2, point_spacing_sec=0.01)
        assert np.max(np.abs([p.velocities for p in points])) <= 0.2 + 1e-9

    def test_no_motion_still_valid(self):
        points = plan_move(START, START, np.full(6, 2.0), min_duration_sec=0.5)
        assert points[-1].time_from_start == pytest.approx(0.5)
        np.testing.assert_array_equal(points[-1].positions, START)


class TestMoveTo:

    def test_reaches_goal(self):
        executor, sim, stop, _ = make_executor()
        result = executor.move_to(GOAL)
        assert result.max_goal_error_rad < 1e-9
        assert stop.reasons == [] and not sim.cancelled
        np.testing.assert_array_equal(sim.points[-1].positions, GOAL)

    def test_goal_outside_limits_never_sends(self):
        executor, sim, _, parser = make_executor()
        goal = GOAL.copy()
        goal[2] = parser.position_limits[1][2] + 0.1
        with pytest.raises(JointLimitError, match="elbow_joint"):
            executor.move_to(goal)
        assert sim.points is None

    def test_waypoint_check_refuses_before_sending(self):
        def outside_box(q):
            if q[0] > START[0] + 0.1:
                raise ValueError("tip leaves the workspace box")

        executor, sim, _, _ = make_executor(waypoint_check=outside_box)
        with pytest.raises(ValueError, match="workspace box"):
            executor.move_to(GOAL)
        assert sim.points is None

    def test_stalled_robot_trips_and_cancels(self):
        sim = Sim()
        sim.robot_stalls = True
        executor, _, stop, _ = make_executor(sim)
        with pytest.raises(TrackingError):
            executor.move_to(GOAL)
        assert sim.cancelled and len(stop.reasons) == 1

    def test_controller_failure_stops(self):
        sim = Sim()
        sim.succeeds = False
        executor, _, stop, _ = make_executor(sim)
        with pytest.raises(CalibrationAbortedError, match="PATH_TOLERANCE_VIOLATED"):
            executor.move_to(GOAL)
        assert len(stop.reasons) == 1

    def test_never_finishing_stops(self):
        sim = Sim()
        sim.finishes = False
        executor, _, stop, _ = make_executor(sim, finish_slack=1.0, finish_margin_sec=0.5)
        with pytest.raises(CalibrationAbortedError, match="didn't finish"):
            executor.move_to(GOAL)
        assert sim.cancelled

    def test_missing_reference_stops(self):
        sim = Sim()
        sim.reports_reference = False
        executor, _, _, _ = make_executor(sim, reference_timeout_sec=0.2)
        with pytest.raises(CalibrationAbortedError, match="no reference"):
            executor.move_to(GOAL)
        assert sim.cancelled

    def test_lost_feedback_mid_move_stops_the_robot(self):
        sim = Sim()
        sim.feedback_lost_after = 0.1
        executor, _, stop, _ = make_executor(sim)
        with pytest.raises(CalibrationAbortedError, match="lost joint state feedback"):
            executor.move_to(GOAL)
        assert sim.cancelled and len(stop.reasons) == 1

    def test_paused_trajectory_stops_on_the_stall_check(self):
        """a paused UR program: the controller holds its reference and the goal never finishes"""
        class Paused(Sim):
            def _elapsed(self):
                return min(super()._elapsed(), 0.2)

            def done(self):
                return False

        sim = Paused()
        executor, _, stop, _ = make_executor(sim, stall_timeout_sec=0.5)
        with pytest.raises(CalibrationAbortedError, match="hasn't moved for 0.5s"):
            executor.move_to(GOAL)
        assert sim.cancelled and len(stop.reasons) == 1
        assert sim.now < 0.2 + 0.5 + 0.05, "caught by the stall check, not the finish deadline"

    def test_slowed_trajectory_is_not_a_stall(self):
        """speed slider at 0.3: the move takes 3.3x as long, the reference keeps moving and the move completes"""
        class Slowed(Sim):
            def _elapsed(self):
                return 0.3 * super()._elapsed()

        result = make_executor(Slowed(), stall_timeout_sec=0.5)[0].move_to(GOAL)
        assert result.max_goal_error_rad <= 0.005

    def test_robot_status_problem_stops_the_move(self):
        sim = Sim()
        status = {"problem": None}
        original = sim.sleep

        def sleep(seconds):
            original(seconds)
            if sim._elapsed() > 0.1:
                status["problem"] = "safety mode PROTECTIVE_STOP"

        sim.sleep = sleep
        executor, _, stop, _ = make_executor(sim, robot_status=lambda: status["problem"])
        with pytest.raises(CalibrationAbortedError, match="PROTECTIVE_STOP"):
            executor.move_to(GOAL)
        assert sim.cancelled and sim._elapsed() < 0.15 and len(stop.reasons) == 1

    def test_robot_status_problem_before_the_move_sends_nothing(self):
        sim = Sim()
        executor, _, _, _ = make_executor(sim, robot_status=lambda: "robot mode POWER_OFF")
        with pytest.raises(CalibrationAbortedError, match="POWER_OFF"):
            executor.move_to(GOAL)
        assert sim.points is None

    def test_finishing_off_goal_stops(self):
        sim = Sim()
        sim.final_offset = np.array([0, 0, 0, 0, 0, 0.02])
        executor, _, _, _ = make_executor(sim, goal_tolerance_rad=0.005)
        with pytest.raises(CalibrationAbortedError, match="wrist_3_joint"):
            executor.move_to(GOAL)

    def test_same_reading_checked_once(self):
        class CountingMonitor:
            tripped = None
            max_state_age_sec = 0.1

            def __init__(self):
                self.checked = []

            def check(self, state, setpoint=None, acceleration=None, command=None, velocity_cap=None):
                self.checked.append(state.received_at)

            def stop(self, reason):
                raise CalibrationAbortedError(reason, 309)

        class SlowFeedback(Sim):
            def sleep(self, seconds):
                # a new joint state only every third poll
                self.now += DT
                if round(self.now / DT) % 3 == 0:
                    self.received_at = self.now

        monitor = CountingMonitor()
        executor, _, _, _ = make_executor(SlowFeedback(), monitor=monitor)
        executor.move_to(GOAL)
        during_move = monitor.checked[:-1]
        assert len(during_move) == len(set(during_move))


class TestJointTrajectoryTransport:

    @pytest.fixture
    def ros(self, monkeypatch):
        state = {"server_up": True, "accepted": True, "goals": [], "cancelled": 0, "callbacks": {}, "answers": True,
                 "pending": []}

        class Future:
            def __init__(self, value, done=True):
                self.value, self._done, self.callbacks = value, done, []

            def done(self):
                return self._done

            def result(self):
                return self.value if self._done else None

            def add_done_callback(self, callback):
                if self._done:
                    callback(self)
                else:
                    self.callbacks.append(callback)

            def resolve(self):
                self._done = True
                for callback in self.callbacks:
                    callback(self)

        class GoalHandle:
            def __init__(self):
                self.accepted = state["accepted"]

            def get_result_async(self):
                return Future(SimpleNamespace(result=SimpleNamespace(error_code=0, error_string="")))

            def cancel_goal_async(self):
                state["cancelled"] += 1

        class ActionClient:
            def __init__(self, node, action_type, name):
                state["action"] = name

            def wait_for_server(self, timeout_sec):
                return state["server_up"]

            def send_goal_async(self, goal):
                state["goals"].append(goal)
                future = Future(GoalHandle(), done=state["answers"])
                state["pending"].append(future)
                return future

            def destroy(self):
                pass

        class Node:
            def create_subscription(self, msg_type, topic, callback, qos):
                state["callbacks"][topic] = callback
                return topic

            def destroy_subscription(self, subscription):
                pass

        class Message(SimpleNamespace):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        class JointTrajectory:
            def __init__(self):
                self.joint_names, self.points = [], []

        FollowJointTrajectory = SimpleNamespace(Goal=Message, Result=SimpleNamespace(SUCCESSFUL=0))
        modules = {
            "rclpy.action": SimpleNamespace(ActionClient=ActionClient),
            "control_msgs": SimpleNamespace(),
            "control_msgs.action": SimpleNamespace(FollowJointTrajectory=FollowJointTrajectory),
            "control_msgs.msg": SimpleNamespace(JointTrajectoryControllerState=object),
            "trajectory_msgs": SimpleNamespace(),
            "trajectory_msgs.msg": SimpleNamespace(JointTrajectory=JointTrajectory, JointTrajectoryPoint=Message),
            "builtin_interfaces": SimpleNamespace(),
            "builtin_interfaces.msg": SimpleNamespace(Duration=Message),
        }
        monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace())
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        return Node(), state

    def test_unreachable_server(self, ros):
        node, state = ros
        state["server_up"] = False
        with pytest.raises(ControllerDriversUnreachableError, match="scaled_joint_trajectory_controller"):
            JointTrajectoryTransport(node, "scaled_joint_trajectory_controller", UR_JOINTS).connect()

    def test_sends_trajectory_in_urdf_order(self, ros):
        node, state = ros
        transport = JointTrajectoryTransport(node, "/scaled_joint_trajectory_controller/", UR_JOINTS)
        points = plan_move(START, GOAL, np.full(6, 2.0))
        transport.send(points)
        assert state["action"] == "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        [goal] = state["goals"]
        assert goal.trajectory.joint_names == UR_JOINTS
        last = goal.trajectory.points[-1]
        assert last.positions == GOAL.tolist()
        assert last.time_from_start.sec * 1e9 + last.time_from_start.nanosec == pytest.approx(points[-1].time_from_start * 1e9)
        assert 0 <= last.time_from_start.nanosec < 1_000_000_000
        assert transport.done() and transport.result() == (True, "error_code 0")

    def test_rejected_goal(self, ros):
        node, state = ros
        state["accepted"] = False
        with pytest.raises(ControllerDriversUnreachableError, match="rejected"):
            JointTrajectoryTransport(node, "arm_controller", UR_JOINTS).send(plan_move(START, GOAL, np.full(6, 2.0)))

    @pytest.mark.parametrize("field", ["reference", "desired"])
    def test_reference_is_reordered(self, ros, field):
        node, state = ros
        transport = JointTrajectoryTransport(node, "arm_controller", UR_JOINTS)
        transport.connect()
        reversed_names = UR_JOINTS[::-1]
        point = SimpleNamespace(positions=list(range(6)), accelerations=[])
        state["callbacks"]["/arm_controller/controller_state"](SimpleNamespace(joint_names=reversed_names, **{field: point}))
        reference = transport.reference()
        np.testing.assert_array_equal(reference.positions, [5, 4, 3, 2, 1, 0])
        assert reference.accelerations is None

    def test_a_rejected_goal_never_ran(self, ros):
        node, state = ros
        state["accepted"] = False
        with pytest.raises(ControllerDriversUnreachableError) as error:
            JointTrajectoryTransport(node, "arm_controller", UR_JOINTS).send(plan_move(START, GOAL, np.full(6, 2.0)))
        assert error.value.goal_may_run is False

    def test_a_goal_accepted_after_the_wait_is_cancelled_on_arrival(self, ros):
        node, state = ros
        state["answers"] = False
        transport = JointTrajectoryTransport(node, "arm_controller", UR_JOINTS, timeout_sec=0.05)
        with pytest.raises(ControllerDriversUnreachableError, match="accept the goal") as error:
            transport.send(plan_move(START, GOAL, np.full(6, 2.0)))
        assert getattr(error.value, "goal_may_run", True)
        assert state["cancelled"] == 0
        state["pending"][-1].resolve()               # the controller accepts it late
        assert state["cancelled"] == 1

    def test_feedback_is_reordered_and_expires(self, ros, monkeypatch):
        node, state = ros
        transport = JointTrajectoryTransport(node, "arm_controller", UR_JOINTS)
        transport.connect()
        assert transport.feedback() is None
        point = SimpleNamespace(positions=list(range(6)), accelerations=[])
        measured = SimpleNamespace(positions=[10, 11, 12, 13, 14, 15])
        state["callbacks"]["/arm_controller/controller_state"](
            SimpleNamespace(joint_names=UR_JOINTS[::-1], reference=point, feedback=measured))
        np.testing.assert_array_equal(transport.feedback(), [15, 14, 13, 12, 11, 10])
        import robot_hal.hardware_communication.motion_executor as module
        later = module.time.monotonic() + 2.0
        monkeypatch.setattr(module.time, "monotonic", lambda: later)
        assert transport.feedback() is None             # a controller_state over a second old isn't trusted

    def test_humble_feedback_is_read_from_actual(self, ros):
        """humble's JointTrajectoryControllerState names the measured point `actual`, jazzy `feedback`"""
        node, state = ros
        transport = JointTrajectoryTransport(node, "arm_controller", UR_JOINTS)
        transport.connect()
        point = SimpleNamespace(positions=list(range(6)), accelerations=[])
        measured = SimpleNamespace(positions=[10, 11, 12, 13, 14, 15])
        state["callbacks"]["/arm_controller/controller_state"](
            SimpleNamespace(joint_names=UR_JOINTS, desired=point, actual=measured))
        np.testing.assert_array_equal(transport.feedback(), [10, 11, 12, 13, 14, 15])

    def test_reference_expires(self, ros, monkeypatch):
        """a frozen controller_state must not keep the path and tracking checks comparing against an old reference"""
        node, state = ros
        transport = JointTrajectoryTransport(node, "arm_controller", UR_JOINTS)
        transport.connect()
        point = SimpleNamespace(positions=list(range(6)), accelerations=[])
        state["callbacks"]["/arm_controller/controller_state"](SimpleNamespace(joint_names=UR_JOINTS, reference=point))
        assert transport.reference() is not None
        import robot_hal.hardware_communication.motion_executor as module
        later = module.time.monotonic() + 1.0
        monkeypatch.setattr(module.time, "monotonic", lambda: later)
        assert transport.reference() is None

    def test_reference_cleared_on_new_goal(self, ros):
        node, state = ros
        transport = JointTrajectoryTransport(node, "arm_controller", UR_JOINTS)
        transport.connect()
        point = SimpleNamespace(positions=list(range(6)), accelerations=[])
        state["callbacks"]["/arm_controller/controller_state"](SimpleNamespace(joint_names=UR_JOINTS, reference=point))
        transport.send(plan_move(START, GOAL, np.full(6, 2.0)))
        assert transport.reference() is None
