
from collections import UserDict
import logging

logger = logging.getLogger(__name__)


def get_timestamp() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ModelDict(UserDict):
    """Helper class to print out model names and keep track of updates

    Args:
        message: description of model
        verbose: print out model names, logging happens regardless
    """

    def __init__(self, message, verbose=True, _timestamp=False, *args, **kwargs):
        self._message = message
        self._verbose = verbose
        self._timestamp = _timestamp  # do no use this inside the class, as it will trigger memoization. Only use outside of class.
        L = (
            len(message)
            if _timestamp is False
            else max(len(message), len(get_timestamp()) + 1)
        )
        self._print_length = min(80, L)
        self._updates = []
        super().__init__(*args, **kwargs)

    def print(self, message):
        if self._timestamp:
            message = f"{message}\n{get_timestamp()}"
        if self._verbose:
            logger.debug('ModelDict: %s', message)

    def __repr__(self):
        return super().__repr__()

    def update(self, *args, **kwargs):
        self._updates.append(args[0])
        if len(self._updates) > 1:  # don't take first update since its the init/default
            self._message += (
                "\n" + "_" * self._print_length + f"\n\nUpdated: {self._updates[-1]}"
            )
        return super().update(*args, **kwargs)
