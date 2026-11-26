"""
finds external measurement hardware without being told it exists. Motor drivers all sit behind ros2_control so one list_controllers
call finds them, metrology hardware has no shared bus like that. What it does have is a discovery mechanism per sensor family that
answers with no setup on our side, the same way list_controllers does:
    GigE Vision cameras        GVCP discovery broadcast, UDP 3956. part of the GigE Vision standard, every compliant camera answers
    Qualisys motion capture    QTM RT discovery broadcast, UDP 22226
    OptiTrack motion capture   NatNet connect broadcast, UDP 1510
    USB cameras                USB Video Class / USB3 Vision interface classes, read from sysfs
Vicon, Leica and FARO have no broadcast discovery, they can only be found once their driver publishes on ros2.

Finding a device here only proves the hardware exists. A camera or mocap server is not a measurement until something turns its data
into poses of the robot, so each device comes back with what has to be set up before it can be used.
"""
import selectors
import socket
import struct
import time
from dataclasses import dataclass
from .._compat import StrEnum
from pathlib import Path
from typing import Callable


class SourceKind(StrEnum):
    VISION = "Vision"
    LASER_TRACKER = "LaserTracker"
    MOTION_CAPTURE = "MotionCapture"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class DetectedDevice:
    kind: SourceKind
    protocol: str
    name: str
    address: str
    setup_hint: str

    def describe(self) -> str:
        return f"{self.name} [{self.protocol}] at {self.address}: {self.setup_hint}"


CAMERA_HINT = (
    "camera found but nothing turns its images into poses. Fix a fiducial (ArUco/AprilTag/ChArUco) to the flange or the cell, "
    "calibrate the camera intrinsics, and run a pose estimator that publishes a PoseStamped or broadcasts a TF frame per marker "
    "(e.g. aruco_ros, apriltag_ros). estimators that only publish arrays or their own message types need a relay to PoseStamped"
)
MOCAP_HINT = (
    "motion capture server found but it isn't streaming to ros2. Define a rigid body on the flange and run a ros2 driver that "
    "publishes it as a PoseStamped or TF frame (e.g. vrpn_mocap). drivers with their own message types need a relay to PoseStamped"
)


def _c_string(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii", errors="replace").strip()


# GigE Vision GVCP (GigE Vision spec, discovery ack payload is the first 0xF8 bytes of the bootstrap registers)
GVCP_PORT = 3956
_GVCP_DISCOVERY_CMD = 0x0002
_GVCP_DISCOVERY_ACK = 0x0003
_GVCP_ACK_PAYLOAD_LEN = 0xF8


def build_gvcp_discovery(_response_port: int = 0, req_id: int = 1) -> bytes:
    # key 0x42, flags 0x11 = acknowledge required + allow broadcast ack (camera may be on another subnet)
    return struct.pack(">BBHHH", 0x42, 0x11, _GVCP_DISCOVERY_CMD, 0, req_id)


def parse_gvcp_discovery_ack(data: bytes, sender: str) -> DetectedDevice | None:
    if len(data) < 8 + _GVCP_ACK_PAYLOAD_LEN:
        return None
    status, answer, _length, _ack_id = struct.unpack_from(">HHHH", data)
    if status != 0 or answer != _GVCP_DISCOVERY_ACK:
        return None
    payload = data[8:]
    manufacturer = _c_string(payload[0x48:0x68])
    model = _c_string(payload[0x68:0x88])
    serial = _c_string(payload[0xD8:0xE8])
    ip = socket.inet_ntoa(payload[0x24:0x28])

    name = " ".join(part for part in (manufacturer, model) if part) or "GigE Vision camera"
    if serial:
        name += f" (serial {serial})"
    return DetectedDevice(SourceKind.VISION, "GigE Vision", name, ip if ip != "0.0.0.0" else sender, CAMERA_HINT)


# Qualisys QTM RT protocol. header is size + type, little endian, the ports are network byte order
QTM_DISCOVERY_PORT = 22226
_QTM_PACKET_COMMAND = 1
_QTM_PACKET_DISCOVER = 7


def build_qtm_discover(response_port: int) -> bytes:
    return struct.pack("<II", 10, _QTM_PACKET_DISCOVER) + struct.pack(">H", response_port)


def parse_qtm_discover_response(data: bytes, sender: str) -> DetectedDevice | None:
    # response is "<host>, QTM <version>, <n> cameras\0" followed by the RT base port
    if len(data) < 11:
        return None
    size, packet_type = struct.unpack_from("<II", data)
    if packet_type not in (_QTM_PACKET_COMMAND, _QTM_PACKET_DISCOVER) or not 11 <= size <= len(data):
        return None
    info = _c_string(data[8:size - 2])
    (base_port,) = struct.unpack_from(">H", data, size - 2)
    return DetectedDevice(SourceKind.MOTION_CAPTURE, "Qualisys QTM", info or "QTM server", f"{sender}:{base_port}", MOCAP_HINT)


# OptiTrack NatNet. message id + byte count little endian, server info starts with a 256 byte app name and 4 byte version
NATNET_COMMAND_PORT = 1510
_NAT_CONNECT = 0
_NAT_SERVERINFO = 1


def build_natnet_connect(_response_port: int = 0) -> bytes:
    return struct.pack("<HH", _NAT_CONNECT, 0)


def parse_natnet_serverinfo(data: bytes, sender: str) -> DetectedDevice | None:
    if len(data) < 4 + 256 + 4:
        return None
    message_id, _n_bytes = struct.unpack_from("<HH", data)
    if message_id != _NAT_SERVERINFO:
        return None
    app = _c_string(data[4:260]) or "NatNet server"
    version = ".".join(str(b) for b in data[260:264])
    return DetectedDevice(SourceKind.MOTION_CAPTURE, "OptiTrack NatNet", f"{app} {version}", sender, MOCAP_HINT)


@dataclass(frozen=True)
class UdpProbe:
    name: str
    port: int
    build: Callable[[int], bytes]
    parse: Callable[[bytes, str], DetectedDevice | None]


UDP_PROBES = (
    UdpProbe("GigE Vision", GVCP_PORT, build_gvcp_discovery, parse_gvcp_discovery_ack),
    UdpProbe("Qualisys QTM", QTM_DISCOVERY_PORT, build_qtm_discover, parse_qtm_discover_response),
    UdpProbe("OptiTrack NatNet", NATNET_COMMAND_PORT, build_natnet_connect, parse_natnet_serverinfo),
)


def udp_discover(probes=UDP_PROBES, timeout_sec: float = 1.0, broadcast_addresses=("255.255.255.255",)) -> list[DetectedDevice]:
    """
    sends every probe's discovery packet at once, one socket each, and collects replies until timeout_sec.
    a probe whose socket can't broadcast (no network, firewall) is skipped rather than failing the whole discovery.
    """
    selector = selectors.DefaultSelector()
    sockets = []
    try:
        for probe in probes:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sockets.append(sock)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.bind(("", 0))
                sock.setblocking(False)
                packet = probe.build(sock.getsockname()[1])
                for address in broadcast_addresses:
                    sock.sendto(packet, (address, probe.port))
            except OSError:
                continue
            selector.register(sock, selectors.EVENT_READ, probe)

        found: dict[tuple[str, str], DetectedDevice] = {}
        deadline = time.monotonic() + timeout_sec
        while selector.get_map() and (remaining := deadline - time.monotonic()) > 0:
            for key, _ in selector.select(timeout=remaining):
                try:
                    data, (sender, _port) = key.fileobj.recvfrom(4096)
                except OSError:
                    continue
                device = key.data.parse(data, sender)
                if device is not None:
                    found[(device.protocol, device.address)] = device
        return list(found.values())
    finally:
        selector.close()
        for sock in sockets:
            sock.close()


_USB_VIDEO_CLASS = "0e"
_USB_MISC_CLASS = "ef"
_USB3_VISION_SUBCLASS = "05"


def _read_sysfs(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def find_usb_cameras(sysfs_root: Path = Path("/sys/bus/usb/devices")) -> list[DetectedDevice]:
    """interface entries are named "<device>:<config>.<interface>", the class says whether it's a camera"""
    if not sysfs_root.is_dir():
        return []
    found: dict[str, DetectedDevice] = {}
    for interface in sorted(sysfs_root.iterdir()):
        if ":" not in interface.name:
            continue
        interface_class = _read_sysfs(interface / "bInterfaceClass")
        if interface_class == _USB_VIDEO_CLASS:
            protocol = "USB Video Class"
        elif interface_class == _USB_MISC_CLASS and _read_sysfs(interface / "bInterfaceSubClass") == _USB3_VISION_SUBCLASS:
            protocol = "USB3 Vision"
        else:
            continue

        device_name = interface.name.split(":", 1)[0]
        if device_name in found:
            continue
        device = sysfs_root / device_name
        name = " ".join(p for p in (_read_sysfs(device / "manufacturer"), _read_sysfs(device / "product")) if p)
        if not name:
            name = f"USB camera {_read_sysfs(device / 'idVendor')}:{_read_sysfs(device / 'idProduct')}"
        found[device_name] = DetectedDevice(SourceKind.VISION, protocol, name, f"usb {device_name}", CAMERA_HINT)
    return list(found.values())


def discover_devices(probe_network: bool = True, probe_usb: bool = True, timeout_sec: float = 1.0,
                     broadcast_addresses=("255.255.255.255",)) -> list[DetectedDevice]:
    devices = []
    if probe_network:
        devices += udp_discover(timeout_sec=timeout_sec, broadcast_addresses=broadcast_addresses)
    if probe_usb:
        devices += find_usb_cameras()
    return devices
