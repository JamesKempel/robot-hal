import os
import re

import numpy as np
import pinocchio as pin
import pytest
from robot_hal.errors import IncompatibleDriverError
from robot_hal.hardware_communication.hardware_controller_interface import DriverType
from robot_hal.parser import Parser

URDF_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ur20.urdf")
UR_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
Q = np.array([0.3, -1.2, 1.1, -0.8, 1.4, 0.5])


@pytest.fixture(scope="module")
def parser():
    return Parser(URDF_PATH)


@pytest.fixture(scope="module")
def fixed():
    """the URDF as a plain fixed base model, what the floating base model has to agree with when bolted down"""
    model = pin.buildModelFromUrdf(URDF_PATH)
    return model, model.createData()


@pytest.fixture
def pos_velo(parser):
    return parser.generate_formulas(DriverType.POS_VELO)


@pytest.fixture
def torque(parser):
    return parser.generate_formulas(DriverType.TORQUE)


class TestKinematics:

    def test_tip_defaults_to_tool0(self, pos_velo):
        assert pos_velo.tip_frame == "tool0"

    def test_forward_kinematics_matches_fixed_base_model(self, fixed, pos_velo):
        model, data = fixed
        pin.framesForwardKinematics(model, data, Q)
        expected = data.oMf[model.getFrameId("tool0")]
        assert np.allclose(pos_velo.forward_kinematics(Q).homogeneous, expected.homogeneous)

    def test_explicit_tip_frame(self, parser):
        formulas = parser.generate_formulas(DriverType.POS_VELO, tip_frame="flange")
        assert formulas.tip_frame == "flange"
        assert not np.allclose(formulas.forward_kinematics(Q).homogeneous,
                               formulas.forward_kinematics(Q, "tool0").homogeneous)

    def test_forward_kinematics_returns_a_copy(self, pos_velo):
        # data is shared between calls, an earlier result must not change when the next call reuses it
        first = pos_velo.forward_kinematics(Q)
        expected = first.homogeneous.copy()
        pos_velo.forward_kinematics(np.zeros(6))
        assert np.allclose(first.homogeneous, expected)

    def test_unknown_tip_frame(self, parser):
        with pytest.raises(ValueError, match="gripper_tip"):
            parser.generate_formulas(DriverType.POS_VELO, tip_frame="gripper_tip")

    def test_wrong_q_shape(self, pos_velo):
        with pytest.raises(ValueError, match="6 entries"):
            pos_velo.forward_kinematics(np.zeros(5))


class TestFloatingBase:

    def test_model_has_a_free_flyer_base(self, parser):
        assert parser.model.joints[1].shortname() == "JointModelFreeFlyer"
        assert parser.model.nq == 13 and parser.model.nv == 12

    def test_base_is_not_a_joint(self, parser):
        assert parser.joint_names == UR_JOINTS

    def test_ceiling_mount_flips_gravity(self, torque):
        ceiling = Parser(URDF_PATH, base_pose=pin.SE3(pin.utils.rpyToMatrix(np.pi, 0.0, 0.0), np.zeros(3)))
        flipped = ceiling.generate_formulas(DriverType.TORQUE)
        # gravity-only effort (no velocity/acceleration) flips sign when the mount is upside down
        assert np.allclose(flipped.command(Q).efforts, -torque.command(Q).efforts)

    def test_base_pose_moves_forward_kinematics(self, pos_velo):
        mount = pin.SE3(pin.utils.rpyToMatrix(0.0, 0.0, 0.7), np.array([1.0, -2.0, 0.5]))
        moved = Parser(URDF_PATH, base_pose=mount.homogeneous).generate_formulas(DriverType.POS_VELO)
        assert np.allclose(moved.forward_kinematics(Q).homogeneous, (mount * pos_velo.forward_kinematics(Q)).homogeneous)

    def test_model_configuration_round_trip(self, parser):
        config = parser.to_model_configuration(Q)
        assert config.shape == (13,) and np.allclose(config[7:], Q)
        assert np.allclose(config[:7], pin.SE3ToXYZQUAT(pin.SE3.Identity()))


class TestJointTypes:

    def test_continuous_joint(self, tmp_path, pos_velo):
        # wrist_3 as continuous: same kinematics, but two model entries (cos, sin) and no position limit
        path = urdf_variant(tmp_path, "wrist_3_joint", "continuous")
        parser = Parser(str(path))
        formulas = parser.generate_formulas(DriverType.POS_VELO)
        assert parser.model.nq == 14 and len(parser.joint_names) == 6
        assert np.allclose(formulas.forward_kinematics(Q).homogeneous, pos_velo.forward_kinematics(Q).homogeneous)
        q = Q.copy()
        q[5] = 20.0  # past +-2pi, fine for a continuous joint
        formulas.command(q)
        lower, upper = parser.position_limits
        assert lower[5] == -np.inf and upper[5] == np.inf and np.isfinite(lower[:5]).all()

    def test_multi_axis_joint_refused(self, tmp_path):
        path = urdf_variant(tmp_path, "elbow_joint", "planar")
        with pytest.raises(ValueError, match="elbow_joint"):
            Parser(str(path))


def urdf_variant(tmp_path, joint, joint_type):
    with open(URDF_PATH) as f:
        urdf = f.read()
    urdf, count = re.subn(rf'<joint name="{joint}" type="[^"]*"', f'<joint name="{joint}" type="{joint_type}"', urdf)
    assert count == 1
    path = tmp_path / f"{joint}_{joint_type}.urdf"
    path.write_text(urdf)
    return path


class TestTorqueCommand:

    def test_holding_still_commands_gravity(self, torque):
        command = torque.command(Q)
        assert command.positions is None and command.velocities is None
        assert command.efforts is not None and np.all(np.isfinite(command.efforts))

    def test_motion_adds_inertial_torque(self, torque):
        a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        at_rest = torque.command(Q).efforts
        in_motion = torque.command(Q, np.zeros(6), a).efforts
        assert not np.allclose(at_rest, in_motion)

    def test_effort_over_limit(self, torque):
        with pytest.raises(ValueError, match="effort"):
            torque.command(Q, np.zeros(6), np.full(6, 500.0))

    def test_massless_urdf_rejected(self, tmp_path):
        with open(URDF_PATH) as f:
            urdf = re.sub(r'<mass value="[^"]*"', '<mass value="0"', f.read())
        path = tmp_path / "massless.urdf"
        path.write_text(urdf)
        parser = Parser(str(path))
        with pytest.raises(IncompatibleDriverError, match="no mass"):
            parser.generate_formulas(DriverType.TORQUE)
        # position drivers don't need inertias
        parser.generate_formulas(DriverType.POS_VELO)


class TestPosVeloCommand:

    def test_passes_targets_through(self, pos_velo):
        v = np.full(6, 0.1)
        command = pos_velo.command(Q, v)
        assert np.allclose(command.positions, Q) and np.allclose(command.velocities, v)
        assert command.efforts is None

    def test_position_only(self, pos_velo):
        assert pos_velo.command(Q).velocities is None

    def test_position_outside_limits(self, pos_velo):
        q = Q.copy()
        q[2] = 3.5  # elbow is limited to +-pi
        with pytest.raises(ValueError, match="elbow_joint"):
            pos_velo.command(q)

    def test_velocity_over_limit(self, pos_velo):
        with pytest.raises(ValueError, match="velocity"):
            pos_velo.command(Q, np.full(6, 5.0))


def test_formulas_before_generate_raises():
    with pytest.raises(RuntimeError):
        Parser(URDF_PATH).formulas


def test_missing_urdf_raises():
    with pytest.raises(FileNotFoundError):
        Parser("/no/such/file.urdf")
