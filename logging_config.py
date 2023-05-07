import logging

import coloredlogs


def setup_logger(logger_name, log_file, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create file handler
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s")
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setLevel(level)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    coloredlogs.install(
        level=level,
        logger=logger,
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
    )
    # logger.addHandler(console_handler)

    return logger


logger = setup_logger("satnbot", "satnbot.log", level=logging.DEBUG)
