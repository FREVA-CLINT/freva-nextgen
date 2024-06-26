"""Define the logging utility"""

import logging
import warnings
from typing import Any, Union

from rich.console import Console
from rich.logging import RichHandler

try:
    from IPython import get_ipython
except ImportError:
    get_ipython = lambda: None  # noqa: E731


APP_NAME = "freva-databrowser"
logger_format = logging.Formatter(
    "%(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

logger_stream_handle = RichHandler(
    rich_tracebacks=True,
    show_path=False,
    markup=True,
    log_time_format="[%Y-%m-%dT%H:%M:%S]",
    console=Console(soft_wrap=False, stderr=True),
)
logger_stream_handle.setLevel(logging.ERROR)
logging.basicConfig(
    format="%(name)s: %(message)s",
    handlers=[logger_stream_handle],
    level=logging.ERROR,
)


class DatabrowserWarning(UserWarning):
    """Custom warning for the databrowser."""


class Logger(logging.Logger):
    """Custom logger to assure that all log handles receive the same level."""

    _is_cli: bool = False
    """Indicate whether or not this logger belongs to a cli process."""

    propagate: bool = True

    def setLevel(self, level: Union[int, str]) -> None:
        super().setLevel(level)
        logger_stream_handle.setLevel(level)

    def set_level(self, level: Union[str, int]) -> None:
        """Set the log level of the logger."""
        self.setLevel(level)

    @property
    def is_cli(self) -> bool:
        """Check if cli flag is set and we are not in an ipython context."""
        is_ipython = getattr(get_ipython(), "kernel", None) is not None
        if self._is_cli and is_ipython is False:
            return True
        return False

    def set_cli(self) -> None:
        """Make this logger a cli logger."""
        self._is_cli = True
        self.set_level(logging.INFO)

    def reset_cli(self) -> None:
        """Reset the cli flag."""
        self._is_cli = False
        self.set_level(logging.ERROR)

    def set_verbosity(self, num: int) -> None:
        """Set the verbosity of a logger."""
        self.set_level(max(logging.INFO - 10 * num, logging.DEBUG))

    def warning(self, msg: object, *args: Any, **kwargs: Any) -> None:
        """Override the warning logger warning."""
        if self.is_cli:
            super().warning(msg, *args, **kwargs)
        else:
            warnings.warn(
                str(msg),
                DatabrowserWarning,
                stacklevel=2,
            )


logging.setLoggerClass(Logger)
