import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from graph_calibration import URDFParser
from robot_hal.errors import (
    CalibrationAbortedError,
    JointLimitError,
    JointMappingError,
    JointStateInvalidError,
    RobotNotStillError,
    SafeStopFailedError,
    SafeStopUnconfiguredError,
    TorqueDriftError,
    TrackingError,
)
from robot_hal.hardware_communication.hardware_controller_interface import DriverType
from robot_hal.hardware_communication.safety_monitor import (
    ControllerSafeStop,
    JointState,
    SafetyMonitor,
    joint_state_from_msg,
)

URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ur20.urdf")
UR_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
Q = np.array([0.3, -1.2, 1.1, -0.8, 1.4, 0.5])
NOW = 100.0


class RecordingStop:
    def __init__(self, fail=False):
        self.reasons = []
        self.fail = fail

    def __call__(self, reason):
        self.reasons.append(reason)
        if self.fail:
            raise RuntimeError("switch_controller not available")


def make_monitor(driver_type=DriverType.TORQUE, stop=None, **kwargs):
    parser = URDFParser(URDF_PATH)
    formulas = parser.generate_initial_formulas(driver_type)
    stop = stop or RecordingStop()
    return SafetyMonitor(parser, stop, clock=lambda: NOW, **kwargs), formulas, stop


def state(q=Q, v=None, efforts=None, age=0.0):
    return JointState(NOW - age, np.asarray(q, dtype=float), np.zeros(6) if v is None else np.asarray(v, dtype=float), efforts)


def check_n(monitor, n, **kwargs):
    for _ in range(n):
        monitor.check(**kwargs)


class TestFeedback:

    def test_healthy_torque_robot_passes(self):
        monitor, formulas, stop = make_monitor()
        check_n(monitor, 10, state=state(efforts=formulas.gravity(Q)), setpoint=Q, command=formulas.command(Q))
        assert stop.reasons == [] and monitor.tripped is None

    def test_stale_state_trips(self):
        monitor, formulas, stop = make_monitor()
        with pytest.raises(JointStateInvalidError, match="old") as error:
            monitor.check(state(efforts=formulas.gravity(Q), age=0.5))
        assert error.value.stopped_safely and len(stop.reasons) == 1

    def test_nan_trips(self):
        monitor, formulas, _ = make_monitor()
        q = Q.copy()
        q[2] = np.nan
        with pytest.raises(JointStateInvalidError, match="NaN"):
            monitor.check(state(q, efforts=formulas.gravity(Q)))

    def test_torque_robot_needs_effort_feedback(self):
        monitor, _, _ = make_monitor()
        with pytest.raises(JointStateInvalidError, match="efforts"):
            monitor.check(state())

    def test_position_robot_ignores_effort_by_default(self):
        # e.g. UR reports motor current, not Nm
        monitor, _, stop = make_monitor(DriverType.POS_VELO)
        check_n(monitor, 10, state=state(efforts=np.full(6, 3.0)))
        assert stop.reasons == []


class TestLimits:

    def test_measured_position_outside_urdf_trips(self):
        monitor, formulas, _ = make_monitor()
        q = Q.copy()
        q[2] = 3.3
        with pytest.raises(JointLimitError, match="elbow_joint"):
            monitor.check(state(q, efforts=formulas.gravity(q)))

    def test_overspeed_trips(self):
        monitor, _, _ = make_monitor(DriverType.POS_VELO)
        with pytest.raises(JointLimitError, match="faster"):
            check_n(monitor, 3, state=state(v=np.full(6, 4.0)))

    def test_single_velocity_spike_tolerated(self):
        """mock hardware: one controller_manager overrun makes one finite difference velocity sample spike"""
        monitor, _, stop = make_monitor(DriverType.POS_VELO)
        monitor.check(state(v=np.full(6, 4.0)))
        check_n(monitor, 5, state=state())
        assert stop.reasons == []


class TestTracking:

    def test_single_spike_tolerated(self):
        monitor, _, stop = make_monitor(DriverType.POS_VELO)
        monitor.check(state(), setpoint=Q + 0.2)
        check_n(monitor, 5, state=state(), setpoint=Q)
        assert stop.reasons == []

    def test_sustained_error_trips(self):
        monitor, _, stop = make_monitor(DriverType.POS_VELO)
        check_n(monitor, 2, state=state(), setpoint=Q + 0.2)
        with pytest.raises(TrackingError, match="setpoint"):
            monitor.check(state(), setpoint=Q + 0.2)
        assert len(stop.reasons) == 1


class TestTorqueDrift:

    def test_model_mismatch_trips(self):
        monitor, formulas, stop = make_monitor()
        heavy = formulas.gravity(Q) * 1.5  # e.g. a payload the URDF doesn't know about
        with pytest.raises(TorqueDriftError, match="wrong torques") as error:
            check_n(monitor, 3, state=state(efforts=heavy))
        assert error.value.stopped_safely
        assert "shoulder_lift_joint" in error.value.message

    def test_drive_not_delivering_command_trips(self):
        monitor, formulas, _ = make_monitor()
        command = formulas.command(Q)
        command = type(command)(efforts=command.efforts + np.array([0, 0, 0, 0, 0, 40.0]))
        with pytest.raises(TorqueDriftError, match="not delivering"):
            check_n(monitor, 3, state=state(efforts=formulas.gravity(Q)), command=command)

    def test_position_robot_opt_in_names_load(self):
        monitor, formulas, _ = make_monitor(DriverType.POS_VELO, check_effort=True)
        with pytest.raises(TorqueDriftError, match="deflection"):
            check_n(monitor, 3, state=state(efforts=formulas.gravity(Q) * 1.5))


    def test_drift_uses_the_mounting(self):
        # a ceiling mounted arm holding still needs the opposite gravity torque, the monitor must not call that drift
        import pinocchio as pin
        parser = URDFParser(URDF_PATH, base_pose=pin.SE3(pin.utils.rpyToMatrix(np.pi, 0.0, 0.0), np.zeros(3)))
        formulas = parser.generate_initial_formulas(DriverType.TORQUE)
        stop = RecordingStop()
        monitor = SafetyMonitor(parser, stop, clock=lambda: NOW)
        check_n(monitor, 10, state=state(efforts=formulas.gravity(Q)), setpoint=Q)
        assert stop.reasons == []


    def test_monitor_follows_an_adopted_model(self):
        import pinocchio as pin
        monitor, formulas, stop = make_monitor()
        parser = monitor._parser
        candidate = parser.copy_model()
        joint_id = candidate.getJointId("shoulder_lift_joint")
        inertia = candidate.inertias[joint_id]
        candidate.inertias[joint_id] = pin.Inertia(inertia.mass * 3.0, inertia.lever, inertia.inertia * 3.0)
        heavier = parser.adopt_model(candidate)
        assert not np.allclose(heavier.gravity(Q), formulas.gravity(Q), atol=10.0)
        check_n(monitor, 10, state=state(efforts=heavier.gravity(Q)), setpoint=Q)
        assert stop.reasons == [] and monitor._model is parser.model


class TestTripBehaviour:

    def test_trip_latches(self):
        monitor, formulas, stop = make_monitor()
        with pytest.raises(JointStateInvalidError):
            monitor.check(state(age=1.0, efforts=formulas.gravity(Q)))
        # even a perfect reading afterwards keeps raising, and the robot isn't stopped twice
        with pytest.raises(JointStateInvalidError):
            monitor.check(state(efforts=formulas.gravity(Q)))
        assert len(stop.reasons) == 1

    def test_failed_stop_says_emergency_stop(self):
        monitor, formulas, _ = make_monitor(stop=RecordingStop(fail=True))
        with pytest.raises(SafeStopFailedError, match="emergency stop") as error:
            monitor.check(state(age=1.0, efforts=formulas.gravity(Q)))
        assert error.value.stopped_safely is False
        assert "old" in error.value.message

    def test_external_stop(self):
        monitor, _, stop = make_monitor()
        with pytest.raises(CalibrationAbortedError, match="marker lost"):
            monitor.stop("marker lost mid babble")
        assert stop.reasons == ["marker lost mid babble"]

    def test_needs_safe_stop(self):
        parser = URDFParser(URDF_PATH)
        parser.generate_initial_formulas(DriverType.POS_VELO)
        with pytest.raises(SafeStopUnconfiguredError):
            SafetyMonitor(parser, None)


class TestStill:

    def test_still_passes(self):
        monitor, _, _ = make_monitor(DriverType.POS_VELO)
        monitor.assert_still(state(v=np.full(6, 1e-4)))

    def test_moving_trips(self):
        monitor, _, stop = make_monitor(DriverType.POS_VELO)
        with pytest.raises(RobotNotStillError, match="wrist_3_joint"):
            monitor.assert_still(state(v=[0, 0, 0, 0, 0, 0.05]))
        assert len(stop.reasons) == 1

    def test_is_still_doesnt_trip(self):
        monitor, _, stop = make_monitor(DriverType.POS_VELO)
        assert monitor.is_still(state(v=[0, 0, 0, 0, 0, 0.05])) is False
        assert monitor.is_still(state(v=np.full(6, 1e-4))) is True
        assert stop.reasons == [] and monitor.tripped is None

    def test_is_still_trips_on_bad_feedback(self):
        monitor, _, stop = make_monitor(DriverType.POS_VELO)
        with pytest.raises(JointStateInvalidError):
            monitor.is_still(state(age=1.0))
        assert len(stop.reasons) == 1


class TestJointStateFromMsg:

    def msg(self, names, positions, velocities=(), efforts=()):
        return SimpleNamespace(name=names, position=positions, velocity=velocities, effort=efforts)

    def test_reorders_and_drops_extra_joints(self):
        names = ["finger_joint"] + UR_JOINTS[::-1]
        result = joint_state_from_msg(self.msg(names, [9.0] + list(Q[::-1])), UR_JOINTS, 1.0)
        assert np.allclose(result.positions, Q)
        assert result.velocities is None and result.efforts is None

    def test_missing_joint(self):
        with pytest.raises(JointMappingError, match="wrist_3_joint"):
            joint_state_from_msg(self.msg(UR_JOINTS[:5], list(Q[:5])), UR_JOINTS, 1.0)

    def test_short_field(self):
        with pytest.raises(JointStateInvalidError, match="effort"):
            joint_state_from_msg(self.msg(UR_JOINTS, list(Q), efforts=[1.0, 2.0]), UR_JOINTS, 1.0)


class TestControllerSafeStop:

    @pytest.fixture
    def ros(self, monkeypatch):
        calls = {"create_node": 0, "create_client": 0, "wait_for_service": 0, "shutdown": 0, "init": 0}
        sent = []
        # ok: a bool for every switch, or a list of answers one per switch. active: what list_controllers reports
        state = {"ok": True, "service_up": True, "context": False, "active": ["arm_controller"]}

        class Request:
            STRICT = 2

            def __init__(self):
                self.timeout = SimpleNamespace(sec=0, nanosec=0)

        class ListRequest:
            pass

        def answer(result):
            return SimpleNamespace(done=lambda: True, result=lambda: result)

        class Client:
            def __init__(self, srv):
                self.srv = srv

            def wait_for_service(self, timeout_sec):
                calls["wait_for_service"] += 1
                return state["service_up"]

            def call_async(self, request):
                if isinstance(request, ListRequest):
                    return answer(SimpleNamespace(controller=[
                        SimpleNamespace(name=name, state="active") for name in state["active"]
                    ] + [SimpleNamespace(name="idle_controller", state="inactive")]))
                sent.append(request)
                ok = state["ok"].pop(0) if isinstance(state["ok"], list) else state["ok"]
                return answer(SimpleNamespace(ok=ok))

        class Node:
            def create_client(self, srv, name):
                calls["create_client"] += 1
                return Client(srv)

            def get_logger(self):
                return SimpleNamespace(error=lambda message: None, warning=lambda message: None)

            def destroy_client(self, client):
                pass

            def destroy_node(self):
                pass

        class Executor:
            def add_node(self, node):
                pass

            def remove_node(self, node):
                pass

            def shutdown(self):
                pass

            def spin_until_future_complete(self, future, timeout_sec):
                pass

        def create_node(name):
            calls["create_node"] += 1
            return Node()

        def init():
            calls["init"] += 1
            state["context"] = True

        def shutdown():
            calls["shutdown"] += 1
            state["context"] = False

        monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace(
            ok=lambda: state["context"], init=init, shutdown=shutdown, create_node=create_node,
        ))
        monkeypatch.setitem(sys.modules, "rclpy.executors", SimpleNamespace(SingleThreadedExecutor=Executor))
        monkeypatch.setitem(sys.modules, "controller_manager_msgs", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "controller_manager_msgs.srv", SimpleNamespace(
            SwitchController=SimpleNamespace(Request=Request), ListControllers=SimpleNamespace(Request=ListRequest),
        ))
        return calls, sent, state

    def test_strict_switch_to_hold(self, ros):
        _, sent, _ = ros
        with ControllerSafeStop(["arm_controller"], ["hold_controller"]) as stop:
            stop("drift")
        [request] = sent
        assert request.deactivate_controllers == ["arm_controller"]
        assert request.activate_controllers == ["hold_controller"]
        assert request.strictness == 2 and request.activate_asap

    def test_trip_does_no_setup_work(self, ros):
        calls, sent, _ = ros
        stop = ControllerSafeStop(["arm_controller"], ["hold_controller"])
        stop.connect()
        before = dict(calls)
        stop("drift")
        stop("drift again")
        assert calls == before and len(sent) == 2
        stop.close()

    def test_missing_service_fails_at_connect(self, ros):
        _, _, state = ros
        state["service_up"] = False
        stop = ControllerSafeStop(["arm_controller"])
        with pytest.raises(SafeStopUnconfiguredError, match="switch_controller"):
            stop.connect()
        assert not stop.connected

    def test_unconnected_stop_raises(self, ros):
        with pytest.raises(RuntimeError, match="never connected"):
            ControllerSafeStop(["arm_controller"])("drift")

    def test_refused_switch_raises(self, ros):
        _, _, state = ros
        state["ok"] = False
        with ControllerSafeStop(["arm_controller"], ["hold_controller"]) as stop:
            with pytest.raises(RuntimeError, match="refused"):
                stop("drift")

    def test_controller_the_driver_already_stopped_is_a_stop(self, ros):
        """UR: controller_stopper deactivates the arm controller when the program stops, the STRICT switch is then refused"""
        _, sent, state = ros
        state["ok"], state["active"] = False, []
        with ControllerSafeStop(["arm_controller"]) as stop:
            stop("program stopped")
        assert len(sent) == 1

    def test_only_the_undone_part_is_retried(self, ros):
        """arm controller already off, the hold controller still has to come on"""
        _, sent, state = ros
        state["ok"], state["active"] = [False, True], []
        with ControllerSafeStop(["arm_controller"], ["hold_controller"]) as stop:
            stop("drift")
        assert [(r.deactivate_controllers, r.activate_controllers) for r in sent] == [
            (["arm_controller"], ["hold_controller"]), ([], ["hold_controller"]),
        ]

    def test_refused_retry_raises(self, ros):
        _, _, state = ros
        state["ok"], state["active"] = False, []
        with ControllerSafeStop(["arm_controller"], ["hold_controller"]) as stop:
            with pytest.raises(RuntimeError, match="refused"):
                stop("drift")

    def test_arm_controller_still_active_after_refusal_raises(self, ros):
        _, sent, state = ros
        state["ok"] = False
        with ControllerSafeStop(["arm_controller"]) as stop:
            with pytest.raises(RuntimeError, match=r"\['arm_controller'\] still active"):
                stop("drift")
        assert len(sent) == 1

    def test_owned_context_is_released(self, ros):
        calls, _, state = ros
        with ControllerSafeStop(["arm_controller"]):
            assert state["context"]
        assert calls["init"] == 1 and calls["shutdown"] == 1 and not state["context"]

    def test_existing_context_left_alone(self, ros):
        calls, _, state = ros
        state["context"] = True
        with ControllerSafeStop(["arm_controller"]):
            pass
        assert calls["shutdown"] == 0 and state["context"]


class TestInterruptedStop:

    def test_ctrl_c_during_the_stop_is_latched(self):
        """ctrl-c or SIGTERM while the stop waits: whether the switch went through is unknown. it must be latched, so the
        handlers further up report that instead of sending a second switch"""
        calls = []

        def interrupted(reason):
            calls.append(reason)
            raise KeyboardInterrupt

        monitor, _, _ = make_monitor(DriverType.POS_VELO, stop=interrupted)
        with pytest.raises(KeyboardInterrupt):
            monitor.stop("marker lost")
        assert isinstance(monitor.tripped, SafeStopFailedError) and "INTERRUPTED" in monitor.tripped.message
        with pytest.raises(SafeStopFailedError):
            monitor.stop("the move was interrupted")
        assert len(calls) == 1


class TestJointStateListener:

    @pytest.fixture
    def listener(self, monkeypatch):
        from robot_hal.hardware_communication import safety_monitor

        monkeypatch.setitem(sys.modules, "sensor_msgs", SimpleNamespace(msg=SimpleNamespace(JointState=object)))
        monkeypatch.setitem(sys.modules, "sensor_msgs.msg", SimpleNamespace(JointState=object))
        node = SimpleNamespace(callbacks=[], destroyed=[])
        node.create_subscription = lambda msg_type, topic, callback, depth: node.callbacks.append(callback) or "subscription"
        node.destroy_subscription = node.destroyed.append
        clock = {"now": 10.0}
        monkeypatch.setattr(safety_monitor.time, "monotonic", lambda: clock["now"])
        return safety_monitor.JointStateListener(node, UR_JOINTS), node, clock

    @staticmethod
    def msg(stamp, q=Q):
        sec = int(stamp)
        return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=int(round((stamp - sec) * 1e9)))),
                               name=UR_JOINTS, position=list(q), velocity=[0.0] * 6, effort=[])

    def test_a_reading_whose_stamp_does_not_advance_is_not_fresh(self, listener):
        """a latched or relayed /joint_states repeats an old stamp: its arrival time must not make it look fresh"""
        listener, node, clock = listener
        node.callbacks[0](self.msg(5.0))
        assert listener.latest().received_at == 10.0
        clock["now"] = 11.0
        node.callbacks[0](self.msg(5.0))          # republished
        node.callbacks[0](self.msg(4.9))          # older
        assert listener.latest().received_at == 10.0
        node.callbacks[0](self.msg(5.1))
        assert listener.latest().received_at == 11.0

    def test_unstamped_readings_are_taken(self, listener):
        listener, node, clock = listener
        node.callbacks[0](self.msg(0.0))
        clock["now"] = 11.0
        node.callbacks[0](self.msg(0.0))
        assert listener.latest().received_at == 11.0

    def test_close_releases_the_subscription(self, listener):
        listener, node, _ = listener
        listener.close()
        listener.close()
        assert node.destroyed == ["subscription"]


class TestBackgroundExecutor:

    def test_a_thread_that_will_not_stop_keeps_the_node(self, monkeypatch):
        import threading

        from robot_hal.hardware_communication.background_executor import BackgroundExecutor

        release = threading.Event()
        removed = []

        class Executor:
            def __init__(self, num_threads):
                pass

            def add_node(self, node):
                pass

            def spin(self):
                release.wait(5.0)

            def shutdown(self):
                pass                                    # a callback that hangs: shutdown doesn't end spin

            def remove_node(self, node):
                removed.append(node)

        monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "rclpy.executors", SimpleNamespace(MultiThreadedExecutor=Executor))
        executor = BackgroundExecutor("node")
        executor.start()
        try:
            with pytest.raises(RuntimeError, match="didn't stop"):
                executor.stop(timeout_sec=0.05)
            assert removed == []
        finally:
            release.set()
        executor.stop()
        assert removed == ["node"]


class TestURRobotStatus:

    @pytest.mark.parametrize("safety, robot, running, expected", [
        (None, None, None, None),                 # nothing received yet: the other checks still catch a stop
        (1, 7, True, None),
        (2, 7, True, None),                       # reduced mode still moves
        (3, 7, True, "PROTECTIVE_STOP"),
        (5, 7, True, "SAFEGUARD_STOP"),
        (7, 7, True, "ROBOT_EMERGENCY_STOP"),
        (1, 3, True, "POWER_OFF"),
        (1, 5, True, "IDLE"),
        (1, 7, False, "External Control program isn't running"),
    ])
    def test_status_problem(self, safety, robot, running, expected):
        from robot_hal.hardware_communication.robot_status import status_problem
        problem = status_problem(safety, robot, running)
        assert problem is None if expected is None else expected in problem

    def test_subscribes_and_reports(self, monkeypatch):
        from robot_hal.hardware_communication.robot_status import URRobotStatus
        monkeypatch.setitem(sys.modules, "std_msgs", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "std_msgs.msg", SimpleNamespace(Bool="Bool"))
        monkeypatch.setitem(sys.modules, "ur_dashboard_msgs", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "ur_dashboard_msgs.msg", SimpleNamespace(RobotMode="RobotMode", SafetyMode="SafetyMode"))
        callbacks, destroyed = {}, []

        class Node:
            def create_subscription(self, msg_type, topic, callback, qos):
                callbacks[topic] = callback
                return topic

            def destroy_subscription(self, subscription):
                destroyed.append(subscription)

        status = URRobotStatus(Node(), "/ur/io_and_status_controller/")
        assert status.problem() is None
        callbacks["/ur/io_and_status_controller/safety_mode"](SimpleNamespace(mode=1))
        callbacks["/ur/io_and_status_controller/robot_mode"](SimpleNamespace(mode=7))
        callbacks["/ur/io_and_status_controller/robot_program_running"](SimpleNamespace(data=True))
        assert status.problem() is None
        callbacks["/ur/io_and_status_controller/safety_mode"](SimpleNamespace(mode=3))
        assert "PROTECTIVE_STOP" in status.problem()
        status.close()
        status.close()
        assert sorted(destroyed) == sorted(callbacks)
