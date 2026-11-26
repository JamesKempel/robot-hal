import socket
import struct
import threading

from robot_hal.hardware_communication.sensor_discovery import (
    SourceKind,
    UdpProbe,
    build_gvcp_discovery,
    build_qtm_discover,
    find_usb_cameras,
    parse_gvcp_discovery_ack,
    parse_natnet_serverinfo,
    parse_qtm_discover_response,
    udp_discover,
)


def gvcp_ack(manufacturer=b"Basler", model=b"acA1920-40gm", serial=b"23456789", ip="192.168.1.20"):
    payload = bytearray(0xF8)
    payload[0x24:0x28] = socket.inet_aton(ip)
    payload[0x48:0x48 + len(manufacturer)] = manufacturer
    payload[0x68:0x68 + len(model)] = model
    payload[0xD8:0xD8 + len(serial)] = serial
    return struct.pack(">HHHH", 0, 0x0003, 0xF8, 1) + bytes(payload)


class TestGigEVision:

    def test_discovery_packet(self):
        assert build_gvcp_discovery() == bytes([0x42, 0x11, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01])

    def test_parse_ack(self):
        device = parse_gvcp_discovery_ack(gvcp_ack(), "10.0.0.1")
        assert device.kind == SourceKind.VISION
        assert device.name == "Basler acA1920-40gm (serial 23456789)"
        assert device.address == "192.168.1.20"

    def test_rejects_short_or_wrong_answer(self):
        assert parse_gvcp_discovery_ack(gvcp_ack()[:100], "10.0.0.1") is None
        bad = bytearray(gvcp_ack())
        bad[2:4] = b"\x00\x05"
        assert parse_gvcp_discovery_ack(bytes(bad), "10.0.0.1") is None


class TestQualisys:

    def test_discover_packet(self):
        assert build_qtm_discover(5000) == struct.pack("<II", 10, 7) + b"\x13\x88"

    def test_parse_response(self):
        info = b"lab-pc, QTM 2023.1, 8 cameras\0"
        data = struct.pack("<II", 8 + len(info) + 2, 1) + info + struct.pack(">H", 22222)
        device = parse_qtm_discover_response(data, "192.168.1.5")
        assert device.kind == SourceKind.MOTION_CAPTURE
        assert device.name == "lab-pc, QTM 2023.1, 8 cameras"
        assert device.address == "192.168.1.5:22222"

    def test_own_broadcast_ignored(self):
        assert parse_qtm_discover_response(build_qtm_discover(5000), "127.0.0.1") is None


class TestNatNet:

    def test_parse_serverinfo(self):
        name = b"Motive".ljust(256, b"\0")
        data = struct.pack("<HH", 1, 264) + name + bytes([3, 1, 0, 0]) + bytes([4, 1, 0, 0])
        device = parse_natnet_serverinfo(data, "192.168.1.7")
        assert device.name == "Motive 3.1.0.0"

    def test_other_message_ignored(self):
        assert parse_natnet_serverinfo(struct.pack("<HH", 7, 260) + bytes(264), "x") is None


class TestUdpDiscover:

    def test_collects_reply(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]

        def respond():
            _, client = server.recvfrom(64)
            server.sendto(gvcp_ack(ip="0.0.0.0"), client)

        thread = threading.Thread(target=respond)
        thread.start()
        probe = UdpProbe("GigE Vision", port, build_gvcp_discovery, parse_gvcp_discovery_ack)
        devices = udp_discover([probe], timeout_sec=0.5, broadcast_addresses=("127.0.0.1",))
        thread.join()
        server.close()
        assert [d.address for d in devices] == ["127.0.0.1"]

    def test_nothing_answers(self):
        probe = UdpProbe("GigE Vision", 9, build_gvcp_discovery, parse_gvcp_discovery_ack)
        assert udp_discover([probe], timeout_sec=0.1, broadcast_addresses=("127.0.0.1",)) == []


class TestUsbCameras:

    def _interface(self, root, name, cls, subclass="00"):
        path = root / name
        path.mkdir()
        (path / "bInterfaceClass").write_text(cls + "\n")
        (path / "bInterfaceSubClass").write_text(subclass + "\n")

    def test_finds_uvc_and_usb3_vision(self, tmp_path):
        (tmp_path / "1-2").mkdir()
        (tmp_path / "1-2" / "manufacturer").write_text("Logitech\n")
        (tmp_path / "1-2" / "product").write_text("BRIO\n")
        self._interface(tmp_path, "1-2:1.0", "0e", "01")
        self._interface(tmp_path, "1-2:1.1", "0e", "02")
        (tmp_path / "2-1").mkdir()
        (tmp_path / "2-1" / "idVendor").write_text("2676\n")
        (tmp_path / "2-1" / "idProduct").write_text("ba02\n")
        self._interface(tmp_path, "2-1:1.0", "ef", "05")
        (tmp_path / "3-1").mkdir()
        self._interface(tmp_path, "3-1:1.0", "03")  # keyboard

        devices = {d.address: d for d in find_usb_cameras(tmp_path)}
        assert set(devices) == {"usb 1-2", "usb 2-1"}
        assert devices["usb 1-2"].name == "Logitech BRIO"
        assert devices["usb 2-1"].protocol == "USB3 Vision"
        assert devices["usb 2-1"].name == "USB camera 2676:ba02"

    def test_missing_sysfs(self, tmp_path):
        assert find_usb_cameras(tmp_path / "nope") == []
