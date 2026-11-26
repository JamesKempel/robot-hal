"""
the tool0 vs tool0_controller mitigation (design §2.8, "what these rules can't cover"): the workspace guard plans on the
nominal URDF and can't catch model error smaller than its clearance, so a wrong robot variant or a joint zero offset still
let the arm leave the box or touch itself in simulation. comparing the URDF's own tool0 against the UR driver's
tool0_controller at the start pose catches that before the first move.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import pytest

# graph_calibration is only present inside the monorepo workspace, not in a
# standalone checkout of this package; skip rather than fail when it's absent.
graph_calibration = pytest.importorskip("graph_calibration")
URDFParser = graph_calibration.URDFParser
from robot_hal.errors import ToolFramePlacementError
from robot_hal.hardware_communication.hardware_controller_interface import DriverType
from robot_hal.hardware_communication.tool_frame_check import (
    check_tool0_against_controller,
    check_tool0_placement,
    read_tf_frame,
)

URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ur20.urdf")
Q = np.array([0.3, -1.2, 1.1, -0.8, 1.4, 0.5])


@pytest.fixture(scope="module")
def formulas():
    return URDFParser(URDF_PATH).generate_initial_formulas(DriverType.POS_VELO)


def true_pose(formulas, tip_frame="tool0", reference_frame="base"):
    return formulas.forward_kinematics(Q, reference_frame).inverse() * formulas.forward_kinematics(Q, tip_frame)


class TestCheckTool0Placement:

    def test_agreement_passes(self, formulas):
        check_tool0_placement(formulas, SimpleNamespace(positions=Q), true_pose(formulas))

    def test_a_placement_error_is_refused(self, formulas):
        off = true_pose(formulas)
        shifted = pin.SE3(off.rotation, off.translation + np.array([0.02, 0.0, 0.0]))  # 20mm, the known failure scale
        with pytest.raises(ToolFramePlacementError, match="20.0mm"):
            check_tool0_placement(formulas, SimpleNamespace(positions=Q), shifted)

    def test_a_zero_offset_sized_rotation_is_refused(self, formulas):
        off = true_pose(formulas)
        twisted = pin.SE3(off.rotation @ pin.exp3(np.array([0.05, 0.0, 0.0])), off.translation)  # the known 0.05rad offset
        with pytest.raises(ToolFramePlacementError):
            check_tool0_placement(formulas, SimpleNamespace(positions=Q), twisted)

    def test_within_tolerance_passes(self, formulas):
        off = true_pose(formulas)
        nudged = pin.SE3(off.rotation, off.translation + np.array([0.001, 0.0, 0.0]))  # 1mm, under the 5mm default
        check_tool0_placement(formulas, SimpleNamespace(positions=Q), nudged)

    def test_tolerance_is_configurable(self, formulas):
        off = true_pose(formulas)
        nudged = pin.SE3(off.rotation, off.translation + np.array([0.001, 0.0, 0.0]))
        with pytest.raises(ToolFramePlacementError):
            check_tool0_placement(formulas, SimpleNamespace(positions=Q), nudged, position_tolerance_m=1e-4)

    def test_error_names_the_frames_and_design_section(self, formulas):
        off = true_pose(formulas)
        shifted = pin.SE3(off.rotation, off.translation + np.array([0.02, 0.0, 0.0]))
        with pytest.raises(ToolFramePlacementError, match=r"tool0.*disagree.*§2\.8"):
            check_tool0_placement(formulas, SimpleNamespace(positions=Q), shifted)


def _transform(child_frame_id, parent_frame_id, translation, rotation):
    x, y, z, w = rotation
    return SimpleNamespace(
        child_frame_id=child_frame_id,
        header=SimpleNamespace(frame_id=parent_frame_id),
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=translation[0], y=translation[1], z=translation[2]),
            rotation=SimpleNamespace(x=x, y=y, z=z, w=w),
        ),
    )


class _FakeTfNode:
    """delivers the given /tf messages, one batch per spin_once call, then nothing"""

    def __init__(self, batches):
        self._batches = list(batches)
        self.subscribed = False

    def create_subscription(self, msg_type, topic, callback, qos):
        assert topic == "/tf"
        self._callback = callback
        self.subscribed = True
        return object()

    def destroy_subscription(self, subscription):
        self.subscribed = False

    def spin_once(self, timeout_sec):
        if self._batches:
            self._callback(SimpleNamespace(transforms=self._batches.pop(0)))


@pytest.fixture
def fake_ros(monkeypatch):
    monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace(spin_once=lambda n, timeout_sec: n.spin_once(timeout_sec)))
    monkeypatch.setitem(sys.modules, "tf2_msgs", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "tf2_msgs.msg", SimpleNamespace(TFMessage=object))


class TestReadTfFrame:

    def test_returns_none_when_the_frame_never_arrives(self, fake_ros):
        node = _FakeTfNode([[_transform("marker_7", "base", (0, 0, 0), (0, 0, 0, 1))]])
        assert read_tf_frame(node, "tool0_controller", timeout_sec=0.05) is None
        assert not node.subscribed

    def test_reads_translation_and_rotation(self, fake_ros):
        node = _FakeTfNode([[_transform("tool0_controller", "base", (0.1, 0.2, 0.3), (0, 0, 0, 1))]])
        pose = read_tf_frame(node, "tool0_controller", timeout_sec=0.05)
        assert np.allclose(pose.translation, [0.1, 0.2, 0.3])
        assert np.allclose(pose.rotation, np.eye(3))

    def test_a_leading_slash_is_stripped(self, fake_ros):
        node = _FakeTfNode([[_transform("/tool0_controller", "/base", (0.1, 0.2, 0.3), (0, 0, 0, 1))]])
        pose = read_tf_frame(node, "tool0_controller", timeout_sec=0.05)
        assert np.allclose(pose.translation, [0.1, 0.2, 0.3])

    def test_the_latest_sample_wins(self, fake_ros):
        node = _FakeTfNode([
            [_transform("tool0_controller", "base", (0.1, 0.0, 0.0), (0, 0, 0, 1))],
            [_transform("tool0_controller", "base", (0.2, 0.0, 0.0), (0, 0, 0, 1))],
        ])
        pose = read_tf_frame(node, "tool0_controller", timeout_sec=0.05)
        assert np.allclose(pose.translation, [0.2, 0.0, 0.0])

    def test_other_frames_are_ignored(self, fake_ros):
        node = _FakeTfNode([[
            _transform("marker_7", "base", (9.0, 9.0, 9.0), (0, 0, 0, 1)),
            _transform("tool0_controller", "base", (0.1, 0.2, 0.3), (0, 0, 0, 1)),
        ]])
        pose = read_tf_frame(node, "tool0_controller", timeout_sec=0.05)
        assert np.allclose(pose.translation, [0.1, 0.2, 0.3])


class TestCheckTool0AgainstController:

    def test_returns_false_when_tool0_controller_never_arrives(self, fake_ros, formulas):
        node = _FakeTfNode([])
        state = SimpleNamespace(positions=Q)
        assert check_tool0_against_controller(node, formulas, state, timeout_sec=0.05) is False

    def test_returns_true_and_passes_on_agreement(self, fake_ros, formulas):
        off = true_pose(formulas)
        t, q = off.translation, pin.Quaternion(off.rotation)
        node = _FakeTfNode([[_transform("tool0_controller", "base", t, (q.x, q.y, q.z, q.w))]])
        state = SimpleNamespace(positions=Q)
        assert check_tool0_against_controller(node, formulas, state, timeout_sec=0.05) is True

    def test_raises_on_disagreement(self, fake_ros, formulas):
        off = true_pose(formulas)
        t, q = off.translation + np.array([0.02, 0.0, 0.0]), pin.Quaternion(off.rotation)
        node = _FakeTfNode([[_transform("tool0_controller", "base", t, (q.x, q.y, q.z, q.w))]])
        state = SimpleNamespace(positions=Q)
        with pytest.raises(ToolFramePlacementError):
            check_tool0_against_controller(node, formulas, state, timeout_sec=0.05)
