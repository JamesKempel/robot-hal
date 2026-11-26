import sys
import threading
from types import SimpleNamespace

from robot_hal.hardware_communication.background_executor import BackgroundExecutor


def test_spins_on_a_thread_until_stopped(monkeypatch):
    events = []

    class Executor:
        def __init__(self, num_threads):
            self._stop = threading.Event()

        def add_node(self, node):
            events.append(("add", node))

        def remove_node(self, node):
            events.append(("remove", node))

        def spin(self):
            events.append(("spin", threading.current_thread().name))
            self._stop.wait(5)

        def shutdown(self):
            self._stop.set()

    monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "rclpy.executors", SimpleNamespace(MultiThreadedExecutor=Executor))
    with BackgroundExecutor("node") as background:
        assert background.running
        background.start()  # a second start is a no-op
    assert not background.running
    assert events[0] == ("add", "node") and events[-1] == ("remove", "node")
    assert ("spin", "calibration_executor") in events
    assert sum(1 for e in events if e[0] == "add") == 1
