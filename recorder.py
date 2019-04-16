from utils import get_formatted_time
import time
import logging
import numpy as np
import json
import uuid

import h5py
import logging.handlers
import os
from config import RecorderConfig

try:
    import cv2
except:
    print("OpenCV not installed! You should not use the monitor!")


class Recorder(object):
    default_config = RecorderConfig().metadata

    def __init__(self, config=None, logger=None, monitoring=False):
        self.created_timestamp = time.time()
        self.created_time = get_formatted_time(self.created_timestamp)
        self.default_config.update(config)
        self.config = self.default_config
        self.logger = logging.getLogger() if not logger else logger
        self.monitoring = monitoring
        if self.monitoring:
            try:
                print("You are using Opencv-python library, version: ", cv2.__version__)
            except:
                raise ValueError("OpenCV not installed!")

        if ("exp_name" in self.config) and (self.config["exp_name"]):
            self.exp_name = self.config["exp_name"]
        else:
            self.exp_name = self.created_time
            self.config["exp_name"] = self.exp_name
        self.save_dir = self.config["save_dir"]

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.dataset_names = self.config["dataset_names"]
        self.initialized_dataset = {k: False for k in self.config["dataset_names"]}
        self.filename = self._get_file_name()

        self.buffer_size = self.config["buffer_size"]
        self.preassigned_buffer_size = self.buffer_size
        self.compress = self.config["compress"] if self.config["compress"] else None
        self.file = None
        self.filemode = None

        self.use_video_writer = self.config["use_video_writer"]
        self.video_writer = None
        self.videofile = None

        if os.path.exists(self.filename):
            self.file = self._get_file('a')
        else:
            self.file = self._get_file('w')
            if self.use_video_writer:
                self.videofile = self.filename.replace("h5", "avi")
                self.logger.info("We will use OpenCV Video Writer to store video at {}.".format(self.videofile))
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.video_writer = cv2.VideoWriter(self.videofile, fourcc, 10, (1280, 960))
                self.dataset_names = list(self.dataset_names)
                self.dataset_names.remove('frame')
        file = self.file

        for ds_name in self.dataset_names:

            if self.initialized_dataset[ds_name]:
                break

            shape = self.config["dataset_shapes"][ds_name]
            shape = (self.preassigned_buffer_size, *shape)
            file.create_dataset(ds_name, shape=shape,
                                dtype=self.config["dataset_dtypes"][ds_name], compression=self.compress,
                                chunks=shape, maxshape=(None, *shape[1:]))

            file.attrs['filename'] = self.filename
            file.attrs['created_timestamp'] = self.created_timestamp
            file.attrs['created_time'] = self.created_time
            file.attrs["video_file_name"] = self.videofile

        config = json.dumps(config)
        file.attrs['config'] = config
        ds_names = json.dumps(self.dataset_names)
        file.attrs["dataset_names"] = ds_names
        timestamp = time.time()
        timen = get_formatted_time(timestamp)

        self.last_modified_timestamp = {k: timestamp for k in self.dataset_names}
        self.last_modified_time = {k: timen for k in self.dataset_names}
        self.buffers = {k: [] for k in self.dataset_names}
        self.accumulated_stored_samples = {k: 0 for k in self.dataset_names}

        info_msg = "{}: HDF5 file {} is ready! With metadata {} and datasets {}".format(self.last_modified_time,
                                                                                        self.filename,
                                                                                        config, ds_names)
        self.logger.info(info_msg)

    def _get_file(self, mode='a'):
        if self.file:
            self.file.close()
        # file = h5py.File(self.filename, mode) #  for cache.
        file = h5py.File(self.filename, mode, rdcc_nbytes=300 * 1024 ** 2)  # 300M for cache.
        for ds_name in self.dataset_names:
            if ds_name in file:
                self.initialized_dataset[ds_name] = True
        self.filemode = mode
        return file

    def add(self, data_dict, force=False):
        assert isinstance(data_dict, dict)
        assert self.filemode is "a" or "w"
        if set(data_dict).add("timestamp") != set(self.dataset_names).add("frame"):
            error_msg = "data_dict is required have same keys as dataset_names, which is {}" \
                        "but only have {}. It may cause the timestamp system mess up!".format(self.dataset_names,
                                                                                              data_dict.keys())
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self._append_to_buffer(np.array((time.time(),)), "timestamp", force)

        # add_to_dataset_flag = False

        for k, data in data_dict.items():
            assert isinstance(data, np.ndarray), "Each entry of data_dict should be a np.ndarray, but get {}.".format(
                type(data))

            if k == 'frame' and self.use_video_writer:
                self.video_writer.write(data)
                continue
            self._append_to_buffer(data, k, force)

        if len(set(self.accumulated_stored_samples.values())) != 1:
            error_msg = "dataset unbalance! The length of each dataset are: {}, but they should be the same!".format(
                self.accumulated_stored_samples)
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if self.monitoring:
            cv2.imshow("Recorder", data_dict["frame"])
            cv2.waitKey(1)

        # if add_to_dataset_flag:
        #     self.accumulated_stored_samples += self.buffer_size

    def _append_to_buffer(self, ndarray, dataset_name, force=False):
        assert isinstance(ndarray, np.ndarray)
        assert ndarray.size == 1 or ndarray.shape[
            0] != 1, "Function add(ndarray) required a single data sample, not a batch!"

        buffer = self.buffers[dataset_name]
        if ndarray is not None:
            buffer.append(ndarray)
        if buffer and (force or len(buffer) == self.buffer_size):
            self.logger.debug(
                "Have collected {} data for dataset {}, prepare to store it. Totally passed {} data.".format(
                    len(buffer), dataset_name,
                    self.accumulated_stored_samples))
            self._append_to_dataset(buffer, dataset_name)
            buffer.clear()
            return True
        return False

    def _append_to_dataset(self, buffer, dataset_name):
        assert isinstance(buffer, list)
        assert isinstance(dataset_name, str)

        now = time.time()
        file = self.file

        ndarray = np.stack(buffer)

        shape = ndarray.shape
        dataset = file[dataset_name]

        current_length = self.accumulated_stored_samples[dataset_name]

        dataset_shape = dataset.shape
        if dataset_shape[0] < current_length + self.buffer_size:
            dataset.resize(dataset.shape[0] + self.preassigned_buffer_size, axis=0)
        self.logger.debug(
            "Prepare to update the dataset {}, in index range [{}, {}]".format(dataset_name, current_length,
                                                                               current_length + shape[0]))

        dataset[current_length: current_length + shape[0]] = ndarray

        self.accumulated_stored_samples[dataset_name] += shape[0]

        self.last_modified_timestamp[dataset_name] = time.time()
        self.last_modified_time[dataset_name] = get_formatted_time(self.last_modified_timestamp[dataset_name])

        dataset.attrs["last_modified_timestamp"] = json.dumps(self.last_modified_timestamp)
        dataset.attrs["last_modified_time"] = json.dumps(self.last_modified_time)

        self.logger.debug("Data has been appended to {} with shape {}. Current dataset {} shape {}.".format(
            dataset.name, ndarray.shape, dataset_name, dataset.shape))
        buffer.clear()

        self.logger.debug("TIMING: recorder take {} seconds to store {} data.".format(time.time() - now, ndarray.shape))

        return dataset_shape

    def _get_file_name(self):
        filename = os.path.join(self.save_dir, "{}.h5".format(self.exp_name))
        return filename

    def read(self):
        # # For testing usage
        ret = {}
        self.file = self._get_file('r')
        file = self.file
        self.logger.debug(
            "Now we have everything in file: {}. self.dataset_names {}.".format(list(file.keys()), self.dataset_names))
        for k in self.dataset_names:
            dset = file[k]
            ret[k] = dset
        # self.file = self._get_file('a')
        return ret

    def display(self):
        self.file = self._get_file('r')
        if self.videofile:
            logging.error("We are using OpenCV for video storage! The video file is in: {}".format(self.videofile))
            return
        frames = self.file["frame"]
        for f in frames:
            cv2.imshow("Replay", f)
            cv2.waitKey(100)
        cv2.destroyAllWindows()
        # self.file = self._get_file('a')

    def close(self):
        if self.monitoring:
            cv2.destroyAllWindows()
        if any([len(buffer) > 0 for buffer in self.buffers.values()]):
            length = len(self.buffers["timestamp"])
            for k, buffer in self.buffers.items():
                if len(buffer) != length:
                    self.logger.warning("The buffer have different length as timestamp! We will clip those excessive.")
                    buffer = buffer[:length]
                self._append_to_dataset(buffer, k)
        if self.video_writer:
            self.video_writer.release()
        self.logger.info("Files has been saved at < {} >.".format(self.filename))
        self.file.close()
        self.logger.debug('Recorder Disconnected. The whole life span of recorder is {} seconds.'.format(
            time.time() - self.created_timestamp))


def build_recorder_process(config, data_queue, log_queue, log_level, monitoring=False):
    qh = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(qh)
    r = Recorder(config, logger, monitoring=monitoring)
    try:
        while True:
            data_dict = data_queue.get()
            if data_dict is None:
                break
            else:
                logger.debug("Recieve: {}".format(data_dict.keys()))
            r.add(data_dict)
    except EOFError:
        logging.error("EOFError happen! The recorder process is killed!")
        raise EOFError
    finally:
        r.close()
    logger.info("Prepare to delete data_queue and log_queue.")
    data_queue.cancel_join_thread()
    log_queue.cancel_join_thread()


def test_generated_data():
    import uuid
    filename = "tmp_{}".format(uuid.uuid4())
    config = {"exp_name": filename,
              "buffer_size": 5,
              "save_dir": 'tmp',
              "compress": "gzip",
              "dataset_names": ("lidar_data", "extra_data", "frame", "timestamp"),
              "dataset_dtypes": {"lidar_data": "uint16", "extra_data": "float32", "frame": "uint8",
                                 "timestamp": "float64"},
              "dataset_shapes": {"lidar_data": (10, 100, 110), "extra_data": (10, 100, 110), "frame": (960, 1280, 3),
                                 "timestamp": (1,)},
              "use_video_writer": True
              }
    r = Recorder(config)
    for _ in range(103):
        data_dict = {k: np.ones([10, 100, 110], dtype=config["dataset_dtypes"][k]) for k in
                     ("lidar_data", "extra_data")}
        data_dict["frame"] = np.random.randint(low=0, high=256, size=(960, 1280, 3), dtype=np.uint8)
        r.add(data_dict)
    filename = r.filename
    r.close()
    return filename


def test_camera_data():
    from camera import setup_camera, close_camera, shot
    filename = "tmp_{}".format(uuid.uuid4())
    config = {"exp_name": filename,
              "buffer_size": 5,
              "save_dir": 'tmp',
              "compress": "gzip",
              "dataset_names": ("lidar_data", "extra_data", "frame", "timestamp"),
              "dataset_dtypes": {"lidar_data": "uint16", "extra_data": "float32", "frame": "uint8",
                                 "timestamp": "float64"},
              "dataset_shapes": {"lidar_data": (30600,), "extra_data": (10, 100, 110), "frame": (960, 1280, 3),
                                 "timestamp": (1,)},
              "use_video_writer": True
              }
    cam = setup_camera()
    r = Recorder(config, monitoring=True)
    for _ in range(200):
        data_dict = {k: np.ones([10, 100, 110], dtype=config["dataset_dtypes"][k]) for k in ("extra_data",)}
        data_dict["lidar_data"] = np.random.randint(0, 30000, size=(30600,), dtype=np.uint16)
        data_dict["frame"] = shot(cam)
        r.add(data_dict)
    r.close()
    return filename


def test_display_and_read(filename):
    config = {"exp_name": filename,
              "buffer_size": 5,
              "save_dir": 'tmp',
              "compress": False,
              "dataset_names": ("lidar_data", "extra_data", "frame", "timestamp"),
              "dataset_dtypes": {"lidar_data": "uint16", "extra_data": "float32", "frame": "uint8",
                                 "timestamp": "float64"},
              "dataset_shapes": {"lidar_data": (10, 100, 110), "extra_data": (10, 100, 110), "frame": (960, 1280, 3),
                                 "timestamp": (1,)},
              }
    r = Recorder(config, monitoring=True)
    d = r.read()
    print(d)
    r.display()
    r.close()


def test_opencv():
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("tmp/tmpx64.avi", fourcc, 10, (1280, 960))
    now = time.time()
    for i in range(200):
        frame = np.random.randint(low=0, high=256, size=(960, 1280, 3), dtype=np.uint8)
        out.write(frame)
    print(time.time() - now)
    out.release()


if __name__ == '__main__':
    from utils import setup_logger
    import uuid

    log_level = "DEBUG"
    setup_logger(log_level)
    filename = test_camera_data()
    test_display_and_read(filename)
