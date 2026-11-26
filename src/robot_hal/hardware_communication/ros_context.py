"""
one place that starts rclpy, so the safe stop survives ctrl-c and SIGTERM.

rclpy.init() installs its own SIGINT / SIGTERM handlers by default, and those shut the rclpy context down. after that no service
call can be spun to completion, so the safe stop (a controller switch, safety_monitor.ControllerSafeStop) can no longer reach the
controller manager: a ctrl-c mid move would leave the trajectory running, and SIGTERM (docker stop, launch shutdown) didn't even
raise in Python, the move ran on unwatched until the safe stop failed.

so rclpy is started without its signal handlers, and SIGINT / SIGTERM raise KeyboardInterrupt in the main thread instead. the
motion loop catches it, stops the robot through the still-working context, cancels the goal and re-raises.
"""
import signal
import threading

_installed = False


class Shutdown(KeyboardInterrupt):
    """SIGTERM, raised in the main thread so the run can stop the robot before exiting"""


def init_rclpy() -> bool:
    """starts rclpy if it isn't running, without its signal handlers. returns whether this call started it"""
    import rclpy

    install_signal_handlers()
    if rclpy.ok():
        return False
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except ImportError:          # very old rclpy without the option: its handlers stay, documented risk
        rclpy.init()
    return True


def install_signal_handlers() -> None:
    """SIGINT raises KeyboardInterrupt, SIGTERM raises Shutdown, in the main thread. only from the main thread, only once"""
    global _installed
    if _installed or threading.current_thread() is not threading.main_thread():
        return

    def on_sigterm(signum, frame):
        raise Shutdown("SIGTERM received")

    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, on_sigterm)
    _installed = True
