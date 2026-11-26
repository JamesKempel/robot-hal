"""
spins a ros2 node on a background thread for the whole babble, so subscriptions (joint states, the sensor, the trajectory
controller's state) and action futures keep updating while the main thread plans, waits and reduces measurements.

one executor for the run, not one per request: a node can only be in one executor, and spinning it from several places
deadlocks or drops callbacks. the safe stop keeps its own node and executor (safety_monitor.ControllerSafeStop), so a trip never
waits on this thread.
    with BackgroundExecutor(node):
        ...move and measure...
callbacks run on the executor's threads. they only copy messages into buffers, everything else (FK, monitor checks) happens on
the caller's thread.
"""
import threading


class BackgroundExecutor:
    def __init__(self, node, num_threads: int = 2):
        self._node = node
        self._num_threads = num_threads
        self._executor = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        # imported here so the library still imports without a sourced ROS env
        from rclpy.executors import MultiThreadedExecutor

        self._executor = MultiThreadedExecutor(num_threads=self._num_threads)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, name="calibration_executor", daemon=True)
        self._thread.start()

    def stop(self, timeout_sec: float = 2.0) -> None:
        """raises RuntimeError if the spin thread is still running after timeout_sec. the node is then left in the executor:
        removing it, or destroying it afterwards, under a thread that may still run its callbacks is unsafe"""
        if self._executor is None:
            return
        self._executor.shutdown()
        if self._thread is not None:
            self._thread.join(timeout_sec)
            if self._thread.is_alive():
                raise RuntimeError(f"the executor thread didn't stop within {timeout_sec}s, its node can't be released safely")
        self._executor.remove_node(self._node)
        self._executor = self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
