"""
rclpy is started without its signal handlers, so ctrl-c and SIGTERM leave the context up for the safe stop (ros_context.py)
"""
import os
import signal
import sys
import time
from types import SimpleNamespace

import pytest
from robot_hal.hardware_communication import ros_context
from robot_hal.hardware_communication.ros_context import Shutdown, init_rclpy, install_signal_handlers


@pytest.fixture
def handlers(monkeypatch):
    """restore the process's signal handlers afterwards"""
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(ros_context, "_installed", False)
    yield
    for s, handler in saved.items():
        signal.signal(s, handler)


@pytest.fixture
def fake_rclpy(monkeypatch):
    calls = {"init": [], "ok": False}
    options = SimpleNamespace(NO="no", ALL="all")
    rclpy = SimpleNamespace(ok=lambda: calls["ok"], init=lambda **kwargs: calls["init"].append(kwargs))
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.signals", SimpleNamespace(SignalHandlerOptions=options))
    return calls


def test_rclpy_starts_without_its_signal_handlers(handlers, fake_rclpy):
    assert init_rclpy() is True
    assert fake_rclpy["init"] == [{"signal_handler_options": "no"}]


def test_a_running_context_is_left_alone(handlers, fake_rclpy):
    fake_rclpy["ok"] = True
    assert init_rclpy() is False and fake_rclpy["init"] == []


def test_sigterm_raises_in_the_main_thread(handlers):
    install_signal_handlers()
    with pytest.raises(Shutdown):
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(1.0)                     # the handler raises out of the sleep


def test_sigint_raises_keyboard_interrupt(handlers):
    install_signal_handlers()
    with pytest.raises(KeyboardInterrupt):
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(1.0)                     # the handler raises out of the sleep


def test_shutdown_is_a_keyboard_interrupt():
    """so everything that already cleans up on ctrl-c (the motion loop, the session teardown) does it on SIGTERM too"""
    assert issubclass(Shutdown, KeyboardInterrupt)
