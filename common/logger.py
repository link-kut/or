import logging, os, sys
from logging.handlers import RotatingFileHandler

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)


def get_logger(name):
    """
    Args:
        name(str):생성할 log 파일명입니다.

    Returns:
         생성된 logger객체를 반환합니다.
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    if not os.path.exists(os.path.join(PROJECT_HOME, "out", "logs")):
        os.makedirs(os.path.join(PROJECT_HOME, "out", "logs"))

    rotate_handler = RotatingFileHandler(
        os.path.join(PROJECT_HOME, "out", "logs", name + ".log"),
        'a',
        1024 * 1024 * 2,
        5
    )
    # formatter = logging.Formatter(
    #     '[%(levelname)s]-%(asctime)s-%(filename)s:%(lineno)s:%(message)s',
    #     datefmt="%Y-%m-%d %H:%M:%S"
    # )

    formatter = logging.Formatter(
        ''
    )

    rotate_handler.setFormatter(formatter)
    logger.addHandler(rotate_handler)
    return logger