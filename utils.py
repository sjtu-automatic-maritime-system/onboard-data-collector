


import time
import logging
import os

def get_formatted_time():
    return time.strftime('%Y-%m-%d_%H-%M-%S',
                  time.localtime())

def get_formatted_log_file_name():
    return time.strftime('%Y-%m-%d_%H-%M-%S_{}.log',
                  time.localtime())

def setup_logger(config):
    if not os.path.exists("log"):
        os.mkdir("log")
    log_format = "[%(levelname)s]\t%(asctime)s: %(message)s"
    log_level = config.log_level.upper()
    logging.basicConfig(filename="log/" + get_formatted_log_file_name().format("VLPmonitor"),
                        level=log_level,
                        format=log_format)
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger("").addHandler(console)