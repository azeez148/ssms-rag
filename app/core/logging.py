import logging
import sys


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s level=%(levelname)s logger=%(name)s message=%(message)s",
        stream=sys.stdout,
    )
