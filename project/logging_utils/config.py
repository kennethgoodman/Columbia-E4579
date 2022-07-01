#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import logging.handlers
from logging.config import dictConfig

logger = logging.getLogger(__name__)

DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
}


def configure_logging(logfile_path, level, logger_name):
    """
    Initialize logging defaults for Project.

    :param logfile_path: logfile used to the logfile
    :type logfile_path: string

    :param level: level used for logger
    :type level: int

    :param logger_name: name for logger
    :type logger_name: string

    This function does:

    - Assign INFO and DEBUG level to logger file handler and console handler

    """
    dictConfig(DEFAULT_LOGGING)

    default_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s():%(lineno)s] [PID:%(process)d TID:%(thread)d] %("
        "message)s",
        "%d/%m/%Y %H:%M:%S")

    file_handler = logging.handlers.RotatingFileHandler(logfile_path, maxBytes=10485760,backupCount=300, encoding='utf-8')
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    file_handler.setFormatter(default_formatter)
    console_handler.setFormatter(default_formatter)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger