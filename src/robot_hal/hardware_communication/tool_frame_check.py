"""
compares the URDF's own tool0 against the UR driver's own idea of it (design §2.8, "what these rules can't cover"). the
workspace guard plans on the nominal URDF and the bootstrap residual can't see model error that small: 5-20mm placement
errors and 0.05rad joint zero offsets still left the box or touched the robot in simulation, every time rejected afterwards
but only after the arm had already moved there. tcp_pose_broadcaster publishes the controller's own forward kinematics as
tool0_controller on /tf (next to robot_state_publisher, see external_sensor_adapter.py), computed from whatever kinematics
the robot was calibrated with at the factory. comparing it against FK(tool0) on the URDF we're about to move and calibrate
with, at the start pose, catches a wrong robot variant or a joint zero offset before the first move, not after. the
comparison is kept relative to the robot's own 'base' frame rather than the world, so it doesn't need base_pose (where the
robot is mounted in the cell), only whether the chain from 'base' to 'tool0' agrees with the controller's.
"""
import time

import numpy as np
import pinocchio as pin

from ..errors.safety_errors import ToolFramePlacementError, TOOL_FRAME_PLACEMENT

# UR controller that publishes tool0_controller on /tf; its absence from active_controllers means nothing to compare against
TCP_POSE_BROADCASTER = "tcp_pose_broadcaster"


def read_tf_frame(node, child_frame: str, timeout_sec: float = 0.5, spin_period_sec: float = 0.05) -> pin.SE3 | None:
    """
    listens to /tf for timeout_sec and returns the last transform received for child_frame as a pin.SE3, or None if it never
    arrived (not every setup runs the controller that publishes it, this is a best-effort read)
    """
    import rclpy
    from tf2_msgs.msg import TFMessage

    latest = None

    def on_tf(msg):
        nonlocal latest
        for transform in msg.transforms:
            if transform.child_frame_id.lstrip("/") == child_frame:
                latest = transform.transform

    subscription = node.create_subscription(TFMessage, "/tf", on_tf, 100)
    try:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=spin_period_sec)
    finally:
        node.destroy_subscription(subscription)

    if latest is None:
        return None
    t, r = latest.translation, latest.rotation
    return pin.SE3(pin.Quaternion(r.w, r.x, r.y, r.z).matrix(), np.array([t.x, t.y, t.z]))


def check_tool0_placement(formulas, state, controller_pose: pin.SE3, tip_frame: str = "tool0", reference_frame: str = "base",
                           position_tolerance_m: float = 5e-3, orientation_tolerance_rad: float = 0.02) -> None:
    """
    raises ToolFramePlacementError if the URDF's FK of tip_frame relative to reference_frame, at the joint angles in state,
    disagrees with controller_pose (reference_frame -> tip_frame, e.g. the controller's own tool0_controller) by more than
    the tolerance. the two are independent: one from the URDF we're about to calibrate, the other from whatever kinematics
    the robot's own controller holds, so agreement doesn't depend on either being correct about where the robot is in the
    cell, only on whether they agree with each other about the robot's own geometry
    """
    base_to_tip = formulas.forward_kinematics(state.positions, reference_frame).inverse() \
        * formulas.forward_kinematics(state.positions, tip_frame)
    offset = controller_pose.inverse() * base_to_tip
    position_error_m = float(np.linalg.norm(offset.translation))
    orientation_error_rad = float(np.linalg.norm(pin.log3(offset.rotation)))
    if position_error_m > position_tolerance_m or orientation_error_rad > orientation_tolerance_rad:
        raise ToolFramePlacementError(
            f"the URDF's {tip_frame} and the controller's own kinematics disagree by {position_error_m * 1e3:.1f}mm and "
            f"{orientation_error_rad:.4f}rad at the start pose (tolerance {position_tolerance_m * 1e3:.1f}mm / "
            f"{orientation_tolerance_rad:.4f}rad). the URDF may not match the robot variant, or a joint has a zero offset "
            "too large for the workspace guard's clearance to catch (design §2.8)",
            TOOL_FRAME_PLACEMENT.error_code,
        )


def check_tool0_against_controller(node, formulas, state, child_frame: str = "tool0_controller", timeout_sec: float = 0.5,
                                    **kwargs) -> bool:
    """
    best-effort: reads child_frame off /tf for timeout_sec and checks it against the URDF if it arrives. returns whether it
    could check (False when child_frame never arrived). kwargs are check_tool0_placement's (tip_frame, reference_frame,
    position_tolerance_m, orientation_tolerance_rad)
    """
    controller_pose = read_tf_frame(node, child_frame, timeout_sec)
    if controller_pose is None:
        return False
    check_tool0_placement(formulas, state, controller_pose, **kwargs)
    return True
