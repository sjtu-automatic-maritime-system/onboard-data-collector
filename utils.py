


import time
import logging
import os

def get_formatted_time(timestamp=None):
    # assert isinstance(timestamp)
    if not timestamp:
        return time.strftime('%Y-%m-%d_%H-%M-%S',
                      time.localtime())
    else:
        return time.strftime('%Y-%m-%d_%H-%M-%S',
                      time.localtime(timestamp))

def get_formatted_log_file_name():
    return time.strftime('%Y-%m-%d_%H-%M-%S_{}.log',
                  time.localtime())

def setup_logger(log_level='INFO'):
    assert isinstance(log_level, str)
    if not os.path.exists("log"):
        os.mkdir("log")
    log_format = "[%(levelname)s]\t%(asctime)s: %(message)s"
    # if config:
    #     log_level = config.log_level.upper()
    # else:
    #     log_level = "INFO"
    logging.basicConfig(filename="log/" + get_formatted_log_file_name().format("VLPmonitor"),
                        level=log_level.upper(),
                        format=log_format)
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger("").addHandler(console)