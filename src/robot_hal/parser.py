"""
a minimal Pinocchio-backed implementation of the Parser/Formulas protocol documented in README.md ("The Parser
protocol"), for driving MotionExecutor and SafetyMonitor against your own robot.

scope: a single arm bolted to the cell (a fixed base) with single-axis joints (revolute, continuous, prismatic) --
the common case for a ROS 2 arm. it builds the pinocchio model with a floating base (so everything outside this file
still works in plain joint space: one value per moving joint, in URDF order, the same order /joint_states reports)
and converts to and from pinocchio's model space internally.

this is deliberately narrow. graph_calibration's own URDFParser is a separate, fuller implementation (non-fixed base
mounts, model-correction/adoption, multi-frame measurement, external sensor transforms) built for the calibration
pipeline; it doesn't live here because none of that is needed to drive MotionExecutor/SafetyMonitor.
"""
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pinocchio as pin

from .errors.motor_controller_errors import IncompatibleDriverError, INCOMPATIBLEDRIVER
from .errors.safety_errors import JointLimitError, JOINT_LIMIT
from .hardware_communication.hardware_controller_interface import DriverType

# ROS-Industrial's standard flange frame, then the URDF convention UR also uses
_DEFAULT_TIP_FRAMES = ("tool0", "flange")
# pinocchio puts "universe" at 0, the root joint it adds is always next
_BASE_JOINT_ID = 1


@dataclass(frozen=True)
class JointCommand:
    """one command in ros2_control terms. only the interfaces the driver type claims are filled in"""
    positions: np.ndarray | None = None
    velocities: np.ndarray | None = None
    efforts: np.ndarray | None = None


@dataclass(frozen=True)
class Formulas:
    """
    formulas bound to the model. q, v and a are in joint space: one entry per joint in Parser.joint_names.
    forward_kinematics: (q, frame=None) -> SE3 of the frame in the world frame. frame defaults to tip_frame, any
        frame in the URDF works.
    command: (q, v=None, a=None) -> JointCommand for the driver type. raises JointLimitError when the target breaks
        a URDF limit.
    """
    driver_type: DriverType
    tip_frame: str
    forward_kinematics: Callable[..., pin.SE3]
    command: Callable[..., JointCommand]


class Parser:
    """
    builds a pinocchio model from a URDF for an arm bolted to the cell, and the formulas MotionExecutor/SafetyMonitor
    need: forward kinematics and the command for a given DriverType (inverse dynamics, for torque drivers).
    """

    def __init__(self, urdf_file_path: str, base_pose=None):
        """
        base_pose: where the robot's base sits in a gravity aligned world frame (z up), as a pin.SE3 or 4x4 matrix.
            defaults to upright at the origin. a wall or ceiling mounted arm has to pass its mounting rotation, or
            gravity points the wrong way and every torque is wrong.
        """
        self.urdf_file_path = os.path.abspath(urdf_file_path)
        if not os.path.exists(self.urdf_file_path):
            raise FileNotFoundError(f"URDF file not found at: {self.urdf_file_path}")

        self._base_pose = pin.SE3.Identity() if base_pose is None else _as_se3(base_pose)
        self._model = pin.buildModelFromUrdf(self.urdf_file_path, pin.JointModelFreeFlyer())
        self._data = self._model.createData()
        self._formulas: Formulas | None = None
        self._index_joints()

    # read only access

    @property
    def model(self) -> pin.Model:
        """the full pinocchio model, floating base included. its q has model.nq entries, use to_model_configuration()"""
        return self._model

    @property
    def joint_names(self) -> list[str]:
        """the moving joints, in URDF order. pinocchio merges fixed joints away, the base isn't one of them"""
        return list(self._joint_names)

    @property
    def position_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) per joint. continuous joints are unlimited, -inf / inf"""
        return self._lower.copy(), self._upper.copy()

    @property
    def velocity_limits(self) -> np.ndarray:
        """per joint, 0 where the URDF leaves it unset"""
        return self._model.velocityLimit[self._v_idx].copy()

    @property
    def effort_limits(self) -> np.ndarray:
        """per joint, 0 where the URDF leaves it unset"""
        return self._model.effortLimit[self._v_idx].copy()

    @property
    def formulas(self) -> Formulas:
        """the formulas from the last generate_formulas() call"""
        if self._formulas is None:
            raise RuntimeError("generate_formulas() has not been called yet")
        return self._formulas

    # joint space <-> model space

    def to_model_configuration(self, q) -> np.ndarray:
        """joint space positions -> the model's q (nq entries), with the base at base_pose"""
        q = self._joint_vector(q, "q")
        config = self._base_config.copy()
        config[self._bounded_q] = q[self._bounded]
        config[self._unbounded_q] = np.cos(q[self._unbounded])
        config[self._unbounded_q + 1] = np.sin(q[self._unbounded])
        return config

    def to_model_tangent(self, x, kind: str = "velocity") -> np.ndarray:
        """joint space velocities/accelerations -> the model's tangent (nv entries). a fixed base doesn't move"""
        x = self._joint_vector(x, kind)
        tangent = np.zeros(self._model.nv)
        tangent[self._v_idx] = x
        return tangent

    def joint_part(self, x) -> np.ndarray:
        """the joint entries of a model tangent vector (e.g. torques)"""
        return np.asarray(x)[..., self._v_idx].copy()

    # formulas

    def generate_formulas(self, driver_type: DriverType, tip_frame: str | None = None) -> Formulas:
        """
        driver_type: what the drivers take as input, from HardwareControllerInterface.get_driver_type.
        tip_frame: the frame a move's forward kinematics defaults to, and what torque commands are checked around.
            defaults to tool0 (or flange, or the last link of the chain if neither exists). any frame in the URDF
            works when passed explicitly to formulas.forward_kinematics.
        """
        try:
            driver_type = DriverType(driver_type)
        except ValueError:
            raise IncompatibleDriverError(
                f"unknown driver type {driver_type!r}, expected one of {[t.value for t in DriverType]}",
                INCOMPATIBLEDRIVER.error_code,
            ) from None
        if driver_type == DriverType.TORQUE:
            self._check_inertias()

        tip_frame = tip_frame or self._default_tip_frame()
        self._frame_id(tip_frame)

        self._formulas = self._build_formulas(driver_type, tip_frame)
        return self._formulas

    def _build_formulas(self, driver_type: DriverType, tip_frame: str) -> Formulas:
        model, data = self._model, self._data

        def forward_kinematics(q, frame=None):
            fid = self._frame_id(tip_frame if frame is None else frame)
            pin.framesForwardKinematics(model, data, self.to_model_configuration(q))
            return pin.SE3(data.oMf[fid])

        def inverse_dynamics(q, v, a):
            return self.joint_part(pin.rnea(model, data, self.to_model_configuration(q),
                                            self.to_model_tangent(v, "velocity"),
                                            self.to_model_tangent(a, "acceleration")))

        if driver_type == DriverType.TORQUE:
            def command(q, v=None, a=None):
                zeros = np.zeros(len(self._joint_names))
                efforts = inverse_dynamics(q, zeros if v is None else v, zeros if a is None else a)
                self._check_limit("effort", np.abs(efforts), self.effort_limits)
                return JointCommand(efforts=efforts)
        else:
            def command(q, v=None, a=None):
                q = self._joint_vector(q, "q")
                self._check_position_limits(q)
                if v is not None:
                    v = self._joint_vector(v, "v")
                    self._check_limit("velocity", np.abs(v), self.velocity_limits)
                return JointCommand(positions=q, velocities=v)

        return Formulas(driver_type, tip_frame, forward_kinematics, command)

    def _index_joints(self):
        """works out where each joint's values sit in the model's q and v"""
        base = self._model.joints[_BASE_JOINT_ID]
        self._base_v = np.arange(base.idx_v, base.idx_v + base.nv)
        self._base_config = pin.neutral(self._model)
        self._base_config[base.idx_q:base.idx_q + base.nq] = pin.SE3ToXYZQUAT(self._base_pose)

        names, ids, v_idx, lower, upper = [], [], [], [], []
        bounded, bounded_q, unbounded, unbounded_q = [], [], [], []
        for joint_id in range(_BASE_JOINT_ID + 1, self._model.njoints):
            joint, name = self._model.joints[joint_id], self._model.names[joint_id]
            if joint.nv != 1:
                raise ValueError(
                    f"joint '{name}' ({joint.shortname()}) has {joint.nv} degrees of freedom, only single axis joints "
                    "(revolute, continuous, prismatic) are supported. /joint_states can't report a multi axis joint "
                    "as one value"
                )
            i = len(names)
            names.append(name)
            ids.append(joint_id)
            v_idx.append(joint.idx_v)
            if joint.nq == 2:
                # continuous joint, pinocchio stores it as (cos, sin) so it has no position limit
                unbounded.append(i)
                unbounded_q.append(joint.idx_q)
                lower.append(-np.inf)
                upper.append(np.inf)
            else:
                bounded.append(i)
                bounded_q.append(joint.idx_q)
                lower.append(self._model.lowerPositionLimit[joint.idx_q])
                upper.append(self._model.upperPositionLimit[joint.idx_q])

        self._joint_names = names
        self._joint_ids = ids
        self._v_idx = np.array(v_idx, dtype=int)
        self._lower, self._upper = np.array(lower, dtype=float), np.array(upper, dtype=float)
        self._bounded, self._bounded_q = np.array(bounded, dtype=int), np.array(bounded_q, dtype=int)
        self._unbounded, self._unbounded_q = np.array(unbounded, dtype=int), np.array(unbounded_q, dtype=int)

    def _frame_id(self, frame: str) -> int:
        if not self._model.existFrame(frame):
            raise ValueError(f"frame '{frame}' is not in the URDF")
        return self._model.getFrameId(frame)

    def _default_tip_frame(self) -> str:
        for name in _DEFAULT_TIP_FRAMES:
            if self._model.existFrame(name):
                return name
        # no standard name, fall back to the last link of the chain
        bodies = [frame.name for frame in self._model.frames if frame.type == pin.FrameType.BODY]
        return bodies[-1]

    def _check_inertias(self):
        """RNEA on a URDF without inertias returns torques that can't move the robot, catch it before anything moves"""
        massless = [name for name, joint_id in zip(self._joint_names, self._joint_ids)
                    if self._model.inertias[joint_id].mass <= 0.0]
        if massless:
            raise IncompatibleDriverError(
                f"torque drivers need the URDF's link inertias to compute commands, but the links moved by "
                f"{massless} have no mass",
                INCOMPATIBLEDRIVER.error_code,
            )

    def _joint_vector(self, x, name: str) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape != (len(self._joint_names),):
            raise ValueError(f"{name} must have {len(self._joint_names)} entries, got shape {x.shape}")
        return x

    def _check_position_limits(self, q):
        outside = [name for name, value, lo, hi in zip(self._joint_names, q, self._lower, self._upper)
                   if not lo <= value <= hi]
        if outside:
            raise JointLimitError(f"target position is outside the URDF limits for {outside}", JOINT_LIMIT.error_code)

    def _check_limit(self, kind: str, magnitudes, limits):
        # URDFs leave unset limits at 0, those joints aren't checked
        over = [name for name, value, limit in zip(self._joint_names, magnitudes, limits) if 0.0 < limit < value]
        if over:
            raise JointLimitError(f"target {kind} is above the URDF limit for {over}", JOINT_LIMIT.error_code)


def _as_se3(pose) -> pin.SE3:
    if isinstance(pose, pin.SE3):
        return pin.SE3(pose)
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError(f"base_pose must be a pin.SE3 or a 4x4 matrix, got shape {pose.shape}")
    return pin.SE3(pose)
