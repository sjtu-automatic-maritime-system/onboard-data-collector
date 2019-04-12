


import time


def get_formatted_time():
    return time.strftime('%Y-%m-%d_%H-%M-%S',
                  time.localtime())