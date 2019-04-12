import os
import os.path as osp


class BaseConfig(object):

    # 本机的ip地址，即在network界面看到的值。以192.168开头。
    local_ip = "127.0.0.1"

    # route_ip = "192.168.1.1"

    vlp_port = "55010"
    image_port = "55011"


class VLPConfig(BaseConfig):
    def __init__(self):
        super(VLPConfig, self).__init__()

    # 原生VLP输入端口
    vlp_raw_port = 2368
    fake_run_time = True



class ImageConfig(BaseConfig):
    def __init__(self):
        super(ImageConfig, self).__init__()
        if not osp.exists(self.log_dir):
            os.mkdir(self.log_dir)

    image_refresh_interval = 1  # period to refresh in second
    if_record_rawdata = True  # record raw data from lidar
    log_dir = './test'
