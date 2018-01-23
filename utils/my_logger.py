import logging
import logging.config
import os

class FileConfigLogger:
    """
    Simple logger for debugging/testing purposes.
    Reads configuration from logging.conf."
    """

    name = None
    path = None
    level = None
    env_key = None
    logger = None


    def __init__(self, name, path, level, env_key):
        self.name = name
        self.path = path
        if(level=="CRITICAL"):
            self.level=logging.CRITICAL
        elif(level=="ERROR"):
            self.level=logging.ERROR
        elif(level=="WARNING"):
            self.level=logging.WARNING
        elif(level=="INFO"):
            self.level=logging.INFO
        else:
            self.level=logging.DEBUG
        self.env_key = env_key
        self.setupLogging()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)

    def setupLogging(self):
        """Setup logging configuration.
        You can give the path to logging configuration as
        LOG_CFG=dir1/dir2/logging.conf python <thisfile>.py
        """

        if (self.path == None):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path = dir_path + "/logging.conf"
        else:
            path = self.path
        value = os.getenv(self.env_key, None)
        if value:
            print("Reading logging configuration from environment...")
            path = value
        if os.path.exists(path):
            logging.config.fileConfig(path, defaults=None, disable_existing_loggers=False)
        else:
            print("WARNING: Logging configuration not found")
            logging.basicConfig(level=self.level)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)

