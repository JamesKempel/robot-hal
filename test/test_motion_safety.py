"""
the motion loop's run-time safety: what has to stop the robot during or right before a move, and what must not.

each test is one hazard found by the pre-hardware safety audit and the fault-injection runs:
    feedback freezing mid move, the executor thread dying, a stale / NaN / moving / misplaced start, a tripped monitor,
    ctrl-c and SIGTERM mid move, a goal accepted after the wait gave up, the arm faster than planned, the controller leaving the
    planned path, two /joint_states publishers
"""
import numpy as np
import pytest
from robot_hal.errors import (
    CalibrationAbortedError,
    ControllerDriversUnreachableError,
    JointLimitError,
    JointStateInvalidError,
    SafeStopFailedError,
)
from robot_hal.hardware_communication.motion_executor import ControllerReference, _Path
from robot_hal.hardware_communication.ros_context import Shutdown
from robot_hal.hardware_communication.safety_monitor import JointState, SafetyMonitor

from test_motion_executor import DT, GOAL, START, RecordingStop, Sim, make_executor


class FailingStop(RecordingStop):
    def __call__(self, reason):
        super().__call__(reason)
        raise RuntimeError("switch_controller did not respond")


def interrupted(interrupt):
    """a Sim whose sleep raises `interrupt` 0.2 s into a move, as ctrl-c or SIGTERM would in the poll loop"""
    class Interrupted(Sim):
        def sleep(self, seconds):
            super().sleep(seconds)
            if self.sent_at is not None and self._elapsed() > 0.2:
                raise interrupt()
    return Interrupted()


def nothing_sent(sim):
    return sim.points is None


class TestFeedbackDuringTheMove:

    def test_frozen_joint_states_stop_within_the_age_limit(self):
        """the listener keeps returning its last reading when /joint_states stops: nothing new arrives, nothing raises"""
        class Frozen(Sim):
            def sleep(self, seconds):
                self.now += DT
                if self._elapsed() < 0.3:
                    self.received_at = self.now

        sim = Frozen()
        executor, _, stop, _ = make_executor(sim)
        with pytest.raises(CalibrationAbortedError, match="no new joint state"):
            executor.move_to(GOAL)
        assert len(stop.reasons) == 1 and sim.cancelled
        # stopped within the 0.1 s age limit plus a poll, not at the end of the move
        assert sim._elapsed() <= 0.3 + 0.1 + 2 * DT
        assert sim._elapsed() < sim.points[-1].time_from_start

    def test_a_dead_executor_thread_stops_the_move(self):
        sim = Sim()
        alive = {"value": True}
        original = sim.sleep

        def sleep(seconds):
            original(seconds)
            if sim._elapsed() > 0.2:
                alive["value"] = False

        sim.sleep = sleep
        executor, _, stop, _ = make_executor(sim, alive=lambda: alive["value"])
        with pytest.raises(CalibrationAbortedError, match="executor thread"):
            executor.move_to(GOAL)
        assert stop.reasons and sim.cancelled


class TestStartChecks:

    def test_a_tripped_monitor_sends_nothing(self):
        executor, sim, stop, _ = make_executor()
        with pytest.raises(CalibrationAbortedError):
            executor._monitor.stop("earlier trip")
        with pytest.raises(CalibrationAbortedError, match="earlier trip"):
            executor.move_to(GOAL)
        assert nothing_sent(sim) and len(stop.reasons) == 1

    def test_a_stale_start_sends_nothing(self):
        sim = Sim()
        sim.now = 30.0                      # the last reading is from t=0, 30 s ago
        executor, _, stop, _ = make_executor(sim)
        with pytest.raises(JointStateInvalidError, match="old"):
            executor.move_to(GOAL)
        assert nothing_sent(sim) and stop.reasons

    def test_a_nan_start_sends_nothing(self):
        class NaNStart(Sim):
            def latest(self):
                state = super().latest()
                q = state.positions.copy()
                q[2] = np.nan
                return JointState(state.received_at, q, state.velocities)

        sim = NaNStart()
        executor, _, _, _ = make_executor(sim)
        with pytest.raises(JointStateInvalidError, match="NaN"):
            executor.move_to(GOAL)
        assert nothing_sent(sim)

    def test_a_start_that_never_settles_sends_nothing(self):
        class Moving(Sim):
            def latest(self):
                state = super().latest()
                return JointState(state.received_at, state.positions, np.full(6, 0.2))

        sim = Moving()
        executor, _, stop, _ = make_executor(sim)
        with pytest.raises(CalibrationAbortedError, match="still moving"):
            executor.move_to(GOAL)
        assert nothing_sent(sim) and stop.reasons

    def test_a_start_that_settles_is_waited_for(self):
        class Ringing(Sim):
            def latest(self):
                state = super().latest()
                v = np.full(6, 0.2) if self.now < 0.3 else state.velocities
                return JointState(state.received_at, state.positions, v)

        sim = Ringing()
        executor, _, stop, _ = make_executor(sim)
        executor.move_to(GOAL)
        assert sim.sent_at >= 0.3 and not stop.reasons

    def test_disagreeing_with_the_controller_sends_nothing(self):
        """the controller measures the arm somewhere else than /joint_states says: a second publisher or a stale reading"""
        sim = Sim()
        sim.feedback = lambda: START + np.array([0.1, 0, 0, 0, 0, 0])
        executor, _, stop, _ = make_executor(sim)
        with pytest.raises(CalibrationAbortedError, match="disagree"):
            executor.move_to(GOAL)
        assert nothing_sent(sim) and stop.reasons

    def test_a_silent_controller_sends_nothing(self):
        """a transport that can report the controller's view but has nothing recent can't confirm the start: refuse, don't
        skip the comparison"""
        sim = Sim()
        sim.feedback = lambda: None
        executor, _, stop, _ = make_executor(sim)
        with pytest.raises(CalibrationAbortedError, match="hasn't reported"):
            executor.move_to(GOAL)
        assert nothing_sent(sim) and stop.reasons

    def test_no_velocities_sends_nothing(self):
        """without velocities the settle wait, the speed cap and the URDF velocity check would all be off"""
        class NoVelocities(Sim):
            def latest(self):
                state = super().latest()
                return JointState(state.received_at, state.positions, None)

        sim = NoVelocities()
        executor, _, _, _ = make_executor(sim)
        with pytest.raises(JointStateInvalidError, match="no velocities"):
            executor.move_to(GOAL)
        assert nothing_sent(sim)

    def test_slow_acceptance_does_not_count_against_the_reference(self):
        """the reference timeout runs from acceptance: a controller slow to accept can't report a reference before it"""
        class SlowAccept(Sim):
            def send(self, points):
                self.now += 0.5                       # longer than the reference timeout
                self.received_at = self.now
                super().send(points)

            def reference(self):
                # the first controller_state after acceptance comes 0.1 s later, inside the timeout counted from acceptance
                return None if self._elapsed() < 0.1 else super().reference()

        sim = SlowAccept()
        executor, _, stop, _ = make_executor(sim, reference_timeout_sec=0.2)
        executor.move_to(GOAL)
        assert not stop.reasons

    def test_lag_at_the_end_of_the_move_is_waited_for(self):
        """the arm reaches the goal a little after the controller reports done: not a fault"""
        class Lagging(Sim):
            def latest(self):
                state = super().latest()
                if self.done() and self._elapsed() - self.points[-1].time_from_start < 0.1:
                    return JointState(state.received_at, state.positions + 0.02, state.velocities)
                return state

        sim = Lagging()
        executor, _, stop, _ = make_executor(sim)
        result = executor.move_to(GOAL)
        assert not stop.reasons and result.max_goal_error_rad <= 0.005

    def test_agreeing_with_the_controller_moves(self):
        sim = Sim()
        sim.feedback = lambda: START + 0.002
        executor, _, stop, _ = make_executor(sim)
        executor.move_to(GOAL)
        assert not stop.reasons


class TestInterruptions:

    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, Shutdown])
    def test_ctrl_c_or_sigterm_mid_move_stops_then_cancels(self, interrupt):
        sim = interrupted(interrupt)
        executor, _, stop, _ = make_executor(sim)
        with pytest.raises(interrupt):
            executor.move_to(GOAL)
        assert len(stop.reasons) == 1 and "interrupted" in stop.reasons[0] and sim.cancelled

    def test_a_failed_stop_after_ctrl_c_says_emergency_stop(self):
        sim = interrupted(KeyboardInterrupt)
        parser = make_executor(sim)[3]
        monitor = SafetyMonitor(parser, FailingStop(), clock=sim.clock)
        executor, _, _, _ = make_executor(sim, monitor=monitor)
        with pytest.raises(SafeStopFailedError, match="emergency stop"):
            executor.move_to(GOAL)
        assert sim.cancelled

    def test_a_goal_accepted_too_late_is_stopped(self):
        """send() gave up waiting for acceptance, the goal may still be running"""
        sim = Sim()

        def send(points):
            sim.points, sim.sent_at = points, sim.now
            raise ControllerDriversUnreachableError("didn't accept the goal within 5s", 100)

        sim.send = send
        executor, _, stop, _ = make_executor(sim)
        with pytest.raises(ControllerDriversUnreachableError):
            executor.move_to(GOAL)
        assert stop.reasons and sim.cancelled

    def test_a_rejected_goal_needs_no_stop(self):
        sim = Sim()

        def send(points):
            error = ControllerDriversUnreachableError("rejected the trajectory", 100)
            error.goal_may_run = False
            raise error

        sim.send = send
        executor, _, stop, _ = make_executor(sim)
        with pytest.raises(ControllerDriversUnreachableError, match="rejected"):
            executor.move_to(GOAL)
        assert not stop.reasons


class TestTheControllerRunsWhatWasChecked:

    def test_faster_than_planned_trips(self):
        class Fast(Sim):
            def latest(self):
                state = super().latest()
                moving = self.points is not None and not self.done()
                return JointState(state.received_at, state.positions, np.full(6, 1.0) if moving else state.velocities)

        sim = Fast()
        executor, _, stop, _ = make_executor(sim)
        with pytest.raises(JointLimitError, match="faster than the planned move"):
            executor.move_to(GOAL)
        assert stop.reasons and sim.cancelled

    def test_planned_speed_does_not_trip(self):
        class Honest(Sim):
            def latest(self):
                state = super().latest()
                if self.points is None or self.done():
                    return state
                t = self._elapsed()
                v = (self._planned(t + DT) - self._planned(t)) / DT
                return JointState(state.received_at, state.positions, v)

        sim = Honest()
        executor, _, stop, _ = make_executor(sim)
        executor.move_to(GOAL)
        assert not stop.reasons

    def test_a_reference_off_the_planned_path_stops(self):
        """another client's goal, or a controller bug: the arm would follow a path the guard never checked"""
        class Detour(Sim):
            def reference(self):
                reference = super().reference()
                if reference is None or self._elapsed() < 0.2:
                    return reference
                return ControllerReference(reference.positions + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.05]))

            def latest(self):
                state = super().latest()
                reference = self.reference()
                q = state.positions if reference is None else reference.positions
                return JointState(state.received_at, q, state.velocities)

        sim = Detour()
        executor, _, stop, _ = make_executor(sim)
        with pytest.raises(CalibrationAbortedError, match="off the planned path"):
            executor.move_to(GOAL)
        assert stop.reasons and sim.cancelled

    def test_path_progress_must_not_go_back(self):
        path = _Path(np.zeros(6), np.ones(6), 0.01)
        assert path.off_path(np.full(6, 0.5)) is None
        assert path.off_path(np.full(6, 0.6)) is None
        assert "went back" in path.off_path(np.full(6, 0.3))
        assert "off the planned path" in _Path(np.zeros(6), np.ones(6), 0.01).off_path(np.r_[0.5, 0.5, 0.5, 0.5, 0.5, 0.6])
        assert _Path(np.zeros(6), np.zeros(6), 0.01).off_path(np.full(6, 0.001)) is None


class TestJumps:
    """two /joint_states publishers interleave their readings: every other one resets the tracking debounce"""

    def monitor(self):
        parser = make_executor()[3]
        clock = {"now": 0.0}
        stop = RecordingStop()
        return SafetyMonitor(parser, stop, clock=lambda: clock["now"]), clock, stop

    def test_an_impossible_jump_trips(self):
        monitor, clock, stop = self.monitor()
        monitor.check(JointState(0.0, START, np.zeros(6)))
        clock["now"] = 0.002
        with pytest.raises(JointStateInvalidError, match="second node publishing"):
            monitor.check(JointState(0.002, np.zeros(6), np.zeros(6)))
        assert stop.reasons

    def test_real_motion_does_not_trip(self):
        monitor, clock, stop = self.monitor()
        q = START.copy()
        for k in range(1, 200):
            clock["now"] = k * 0.002
            q = q + 0.5 * 0.002                    # 0.5 rad/s on every joint, a fast babble move
            monitor.check(JointState(clock["now"], q, np.full(6, 0.5)))
        assert not stop.reasons

    def test_interleaved_zeros_trip_even_though_each_reading_is_plausible(self):
        monitor, clock, stop = self.monitor()
        with pytest.raises(JointStateInvalidError):
            for k in range(10):
                clock["now"] = k * 0.002
                monitor.check(JointState(clock["now"], START if k % 2 == 0 else np.zeros(6), np.zeros(6)))
        assert stop.reasons
