from recorder import build_recorder_process
from camera import setup_camera, close_camera, shot
from VLP import setup_vlp, close_vlp
from utils import setup_logger
from config import RecorderConfig
import logging
import argparse
import time
import multiprocessing

"""
Example Usages: 
1. Run the data collector for nearly half hour:
    
    python collect_data.py --exp-name 0101-Trimaran-Tracking --timestep 18000

or

    python collect_data.py --exp-name 0101-Trimaran-Tracking -t 18000
    
2. Run the data collector for not pre-defined time duration:

    python collect_data.py --exp-name 0101-Trimaran-Tracking
    
(Note that you should press Ctrl+C to terminate this program!)
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default=None, type=str)
    parser.add_argument("--log-level", default="INFO", type=str)
    parser.add_argument("--timestep", "-t", default=-1, type=int)
    args = parser.parse_args()

    setup_logger(args.log_level)

    recorder_config = RecorderConfig()
    recorder_config.metadata.update({"exp_name": args.exp_name})

    log_queue = multiprocessing.Queue()
    data_queue = multiprocessing.Queue()
    recorder_process = multiprocessing.Process(target=build_recorder_process, args=(recorder_config.metadata, data_queue, log_queue, args.log_level))
    recorder_process.start()

    vlp = setup_vlp()
    cam = setup_camera()
    now = time.time()

    st = now
    cnt = 0
    log_interval = 10

    try:
        logging.info("Start Record Data!")
        while True:
            logging.debug("The {} iteration!".format(cnt))

            lidar_data, extra_data = vlp.update()
            frame = shot(cam)
            data_dict = {"lidar_data": lidar_data, "extra_data": extra_data, "frame": frame}
            data_queue.put(data_dict)

            if cnt%log_interval==0:
                if args.timestep==-1:
                    logging.info("Data processed in frequency {}. Press Ctrl+C to terminate this program!".format(log_interval/(time.time()-now)))
                else:
                    logging.info("Data processed in frequency {}.".format(log_interval/(time.time()-now)))
                now = time.time()
            cnt += 1
            if args.timestep>0 and cnt==args.timestep:
                break
    finally:
        et = time.time()
        logging.info("Recording Finish! It take {} seconds and collect {} data! Average FPS {}.".format(et-st, cnt, cnt/(et-st)))
        close_vlp(vlp)
        close_camera(cam)
        data_queue.put(None)
        recorder_process.join()