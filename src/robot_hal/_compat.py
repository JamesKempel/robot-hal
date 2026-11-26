"""backports for the oldest python we support (3.10, ROS 2 Humble)"""
from enum import Enum

try:
    from enum import StrEnum
except ImportError:  # python < 3.11
    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value
