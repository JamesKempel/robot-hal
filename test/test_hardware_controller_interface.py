import pytest
from robot_hal.hardware_communication.hardware_controller_interface import (
    DriverType,
    HardwareControllerInterface,
    classify_claimed_interfaces,
)
from robot_hal.errors import IncompatibleDriverError

UR_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"]


class TestClassifyClaimedInterfaces:

    def test_position_is_pos_velo(self):
        claimed = [f"{j}/position" for j in UR_JOINTS]
        assert classify_claimed_interfaces(claimed) == DriverType.POS_VELO

    def test_position_and_velocity_is_pos_velo(self):
        claimed = [f"{j}/position" for j in UR_JOINTS] + [f"{j}/velocity" for j in UR_JOINTS]
        assert classify_claimed_interfaces(claimed) == "PosVelo"

    def test_effort_is_torque(self):
        claimed = [f"{j}/effort" for j in UR_JOINTS]
        assert classify_claimed_interfaces(claimed) == DriverType.TORQUE

    def test_ignores_non_joint_interfaces(self):
        claimed = [f"{j}/effort" for j in UR_JOINTS] + ["gpio/standard_digital_output_cmd_0"]
        assert classify_claimed_interfaces(claimed) == DriverType.TORQUE

    def test_joint_filter(self):
        claimed = [f"{j}/position" for j in UR_JOINTS] + ["gripper_joint/effort"]
        assert classify_claimed_interfaces(claimed, set(UR_JOINTS)) == DriverType.POS_VELO

    def test_mixed_raises(self):
        claimed = ["shoulder_pan_joint/position", "elbow_joint/effort"]
        with pytest.raises(IncompatibleDriverError):
            classify_claimed_interfaces(claimed)

    def test_nothing_recognised_raises(self):
        with pytest.raises(IncompatibleDriverError):
            classify_claimed_interfaces(["gpio/standard_digital_output_cmd_0"])


class TestHardwareControllerInterface:

    def test_driver_type_is_cached(self, monkeypatch):
        calls = []

        def fake_request(self):
            calls.append(1)
            return {"effort_controller": [f"{j}/effort" for j in UR_JOINTS]}

        monkeypatch.setattr(HardwareControllerInterface, "_request_active_controllers", fake_request)
        hw = HardwareControllerInterface()
        assert hw.get_driver_type == DriverType.TORQUE
        assert hw.get_driver_type == DriverType.TORQUE
        assert len(calls) == 1
    def test_joint_controllers_only_lists_our_joints(self, monkeypatch):
        monkeypatch.setattr(HardwareControllerInterface, "_request_active_controllers", lambda self: {
            "scaled_joint_trajectory_controller": [f"{j}/position" for j in UR_JOINTS],
            "speed_scaling_state_broadcaster": ["speed_scaling/speed_scaling_factor"],
            "gripper_controller": ["finger_joint/position"],
        })
        hw = HardwareControllerInterface(joint_names=UR_JOINTS)
        hw.refresh_driver_type()
        assert list(hw.joint_controllers()) == ["scaled_joint_trajectory_controller"]
