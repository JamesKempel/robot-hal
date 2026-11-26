"""
class meant to representthe interface for testing hardware controllers to ensure we can communicate with them
and will determine which pinocchio formulas we will use. Will also change entire workflow for the robot once this
is determined and kept in memory/storage as it determines what data we give the robot controllers.
** might change to abstract later **

Detection goes through ros2_control: the controller manager's list_controllers service reports, for every
loaded controller, the command interfaces it has claimed ("<joint>/position", "<joint>/velocity",
"<joint>/effort"). Whatever the active controllers claimed is what the drivers actually take as input.
"""
from .._compat import StrEnum

from ..errors.motor_controller_errors import (
    ControllerDriversUnreachableError,
    DRIVER_UNREACHABLE_ERROR,
    IncompatibleDriverError,
    INCOMPATIBLEDRIVER,
)


class DriverType(StrEnum):
    TORQUE = "Torque"
    POS_VELO = "PosVelo"


# ros2_control calls torque "effort"
_TORQUE_INTERFACES = {"effort"}
_POS_VELO_INTERFACES = {"position", "velocity"}
_JOINT_INTERFACES = _TORQUE_INTERFACES | _POS_VELO_INTERFACES


def classify_claimed_interfaces(claimed_interfaces, joint_names=None) -> DriverType:
    """
    takes the claimed command interfaces ("<joint>/<interface>") of the active controllers and decides the driver type.
    non joint interfaces (gpio, speed scaling, etc.) are ignored. If joint_names is given, only those joints count.
    Mixing torque and pos/velo across joints is not supported, neither is having nothing we recognise.
    """
    found = set()
    for name in claimed_interfaces:
        prefix, _, interface = name.rpartition("/")
        if interface not in _JOINT_INTERFACES:
            continue
        if joint_names is not None and prefix not in joint_names:
            continue
        found.add(interface)

    if not found:
        raise IncompatibleDriverError(
            f"no position, velocity or effort command interfaces claimed by active controllers: {list(claimed_interfaces)}. "
            "check the arm's controller is loaded and active (ros2 control list_controllers)",
            INCOMPATIBLEDRIVER.error_code,
        )
    if found <= _TORQUE_INTERFACES:
        return DriverType.TORQUE
    if found <= _POS_VELO_INTERFACES:
        return DriverType.POS_VELO
    raise IncompatibleDriverError(
        f"active controllers mix torque and position/velocity command interfaces: {sorted(found)}",
        INCOMPATIBLEDRIVER.error_code,
    )


class HardwareControllerInterface:
    """
    ros2 specific, can turn into an abstract class later
    Need to check what controller the hardware uses initially. It can use velo and position or torque
    If anything else we raise an error that prevents library usage
    """
    def __init__(self, node=None, controller_manager: str = "/controller_manager",
                 joint_names=None, timeout_sec: float = 5.0):
        """
        node: optional rclpy node to make the request from. if None a temporary one is created per request.
        controller_manager: namespace of the ros2_control controller manager.
        joint_names: optional joints to restrict detection to (e.g. the URDF's actuated joints).
        timeout_sec: how long to wait for the controller manager before raising ControllerDriversUnreachableError.
        """
        self._node = node
        self._service_name = f"{controller_manager.rstrip('/')}/list_controllers"
        self._joint_names = set(joint_names) if joint_names is not None else None
        self._timeout_sec = timeout_sec
        self._driver_type = None
        self._active_controllers: dict[str, list[str]] = {}

    @property
    def get_driver_type(self) -> DriverType:
        """
        returns the driver type(str). Can be Torque or PosVelo
        queried from ros2 once, then kept in memory. call refresh_driver_type() to query again.
        """
        if self._driver_type is None:
            self._driver_type = self.refresh_driver_type()
        return self._driver_type

    @property
    def node(self):
        return self._node

    @property
    def controller_manager(self) -> str:
        return self._service_name.rsplit("/", 1)[0]

    @property
    def joint_names(self) -> set[str] | None:
        return self._joint_names

    @property
    def active_controllers(self) -> dict[str, list[str]]:
        """active controller name -> the interfaces it claimed, from the last driver type query"""
        return dict(self._active_controllers)

    def joint_controllers(self) -> dict[str, list[str]]:
        """the active controllers that claimed a command interface on one of our joints, these are what a safe stop switches off"""
        found = {}
        for name, claimed in self._active_controllers.items():
            joint_interfaces = [
                interface for interface in claimed
                if interface.rpartition("/")[2] in _JOINT_INTERFACES
                and (self._joint_names is None or interface.rpartition("/")[0] in self._joint_names)
            ]
            if joint_interfaces:
                found[name] = joint_interfaces
        return found

    def refresh_driver_type(self) -> DriverType:
        self._active_controllers = self._request_active_controllers()
        claimed = [interface for interfaces in self._active_controllers.values() for interface in interfaces]
        self._driver_type = classify_claimed_interfaces(claimed, self._joint_names)
        return self._driver_type

    def _request_active_controllers(self) -> dict[str, list[str]]:
        """sends the list_controllers request and returns every active controller with the interfaces it claimed"""
        # imported here so the library still imports without a sourced ROS env
        import rclpy
        from controller_manager_msgs.srv import ListControllers

        from .ros_context import init_rclpy

        owns_context = init_rclpy()
        node = self._node or rclpy.create_node("hardware_controller_interface")
        try:
            client = node.create_client(ListControllers, self._service_name)
            try:
                if not client.wait_for_service(timeout_sec=self._timeout_sec):
                    raise ControllerDriversUnreachableError(
                        f"{self._service_name} not available after {self._timeout_sec}s",
                        DRIVER_UNREACHABLE_ERROR.error_code,
                    )
                future = client.call_async(ListControllers.Request())
                rclpy.spin_until_future_complete(node, future, timeout_sec=self._timeout_sec)
                if not future.done() or future.result() is None:
                    raise ControllerDriversUnreachableError(
                        f"{self._service_name} did not respond within {self._timeout_sec}s",
                        DRIVER_UNREACHABLE_ERROR.error_code,
                    )
            finally:
                node.destroy_client(client)

            return {
                controller.name: list(controller.claimed_interfaces)
                for controller in future.result().controller
                if controller.state == "active"
            }
        finally:
            if self._node is None:
                node.destroy_node()
            if owns_context:
                rclpy.shutdown()
