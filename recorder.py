from utils import get_formatted_time
import time
import logging
import numpy as np
import json

import h5py
import logging.handlers
import os

class Recorder(object):

    def __init__(self, config=None, logger=None):
        self.created_timestamp = time.time()
        self.created_time = get_formatted_time(self.created_timestamp)
        self.config = config
        self.logger = logging.getLogger() if not logger else logger

        if ("exp_name" in self.config) and (self.config["exp_name"]):
            self.exp_name = self.config["exp_name"]
        else:
            self.exp_name = self.created_time
        self.save_dir = self.config["save_dir"]

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.dataset_names = self.config["dataset_names"]
        self.initialized_dataset = {k: False for k in self.config["dataset_names"]}
        self.filename = self._get_file_name()
        self.buffer_size = self.config["buffer_size"]
        self.preassigned_buffer_size = self.buffer_size
        self.compress = self.config["compress"] if self.config["compress"] else None

        self.accumulated_stored_samples = 0
        self.file = self._get_file()
        file = self.file

        for ds_name in self.dataset_names:

            if ds_name in file:
                self.initialized_dataset[ds_name] = True
                continue

            shape = self.config["dataset_shapes"][ds_name]
            shape = (self.preassigned_buffer_size, *shape)
            file.create_dataset(ds_name, shape=shape,
                                dtype=self.config["dataset_dtypes"][ds_name], compression=self.compress,
                                chunks=shape, maxshape=(None, *shape[1:]))

            file.attrs['filename'] = self.filename
            file.attrs['created_timestamp'] = self.created_timestamp
            file.attrs['created_time'] = self.created_time

        config = json.dumps(config)
        file.attrs['config'] = config
        ds_names = json.dumps(self.dataset_names)
        file.attrs["dataset_names"] = ds_names
        timestamp = time.time()
        timen = get_formatted_time(timestamp)

        self.last_modified_timestamp = {k: timestamp for k in self.dataset_names}
        self.last_modified_time = {k: timen for k in self.dataset_names}
        self.buffers = {k: [] for k in self.dataset_names}

        info_msg = "{}: HDF5 file {} is ready! With metadata {} and datasets {}".format(self.last_modified_time,
                                                                                        self.filename,
                                                                                        config, ds_names)
        self.logger.info(info_msg)

    def _get_file(self, mode='a'):
        # file = h5py.File(self.filename, mode) #  for cache.
        file = h5py.File(self.filename, mode, rdcc_nbytes=300 * 1024 ** 2)  # 300M for cache.
        for ds_name in self.dataset_names:
            if ds_name in file:
                self.initialized_dataset[ds_name] = True
        return file

    def add(self, data_dict, force=False):
        assert isinstance(data_dict, dict)
        if len(data_dict.keys()) + 1 != len(self.dataset_names):
            error_msg = "data_dict is required have same keys as dataset_names, which is {}" \
                        "but only have {}. It may cause the timestamp system mess up!".format(self.dataset_names,
                                                                                              data_dict.keys())
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self._append_to_buffer(np.array(time.time()), "timestamp", force)

        add_to_dataset_flag = False

        for k, data in data_dict.items():
            assert isinstance(data, np.ndarray), "Each entry of data_dict should be a np.ndarray, but get {}.".format(
                type(data))
            assert k in self.dataset_names, "Key Error! Key of data_dict should be in {}, but get {}.".format(
                self.dataset_names, k)
            ret = self._append_to_buffer(data, k, force)
            add_to_dataset_flag = add_to_dataset_flag or ret

        if add_to_dataset_flag:
            self.accumulated_stored_samples += self.buffer_size

    def _append_to_buffer(self, ndarray, dataset_name, force=False):
        assert isinstance(ndarray, np.ndarray)
        assert ndarray.size == 1 or ndarray.shape[
            0] != 1, "Function add(ndarray) required a single data sample, not a batch!"

        buffer = self.buffers[dataset_name]
        if ndarray is not None:
            buffer.append(ndarray)
        if buffer and (force or len(buffer) == self.buffer_size):
            self.logger.debug("Have collected {} data, prepare to store it. Totally passed {} data.".format(len(buffer),
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
        self.logger.debug("Current dataset {}, appending data min {}, mean {}, max {}, "
                          "standard derivative {}, shape {}, dtype {}.".format(dataset_name, ndarray.min(),
                                                                               ndarray.mean(),
                                                                               ndarray.max(), ndarray.std(),
                                                                               ndarray.shape,
                                                                               ndarray.dtype))
        shape = ndarray.shape
        dataset = file[dataset_name]
        dataset_shape = dataset.shape

        if dataset_shape[0] <= self.accumulated_stored_samples - self.buffer_size:
            dataset.resize(dataset.shape[0] + self.preassigned_buffer_size, axis=0)

        dataset[self.accumulated_stored_samples: self.accumulated_stored_samples + shape[0]] = ndarray

        self.last_modified_timestamp[dataset_name] = time.time()
        self.last_modified_time[dataset_name] = get_formatted_time(self.last_modified_timestamp[dataset_name])

        dataset.attrs["last_modified_timestamp"] = json.dumps(self.last_modified_timestamp)
        dataset.attrs["last_modified_time"] = json.dumps(self.last_modified_time)

        self.logger.debug("Data has been appended to {} with shape {}. Current dataset {} shape {}.".format(
            dataset.name, ndarray.shape, dataset_name, dataset_shape))
        buffer.clear()

        self.logger.debug("TIMING: recorder take {} seconds to store {} data.".format(time.time() - now, ndarray.shape))

        return dataset_shape

    def _get_file_name(self):
        filename = os.path.join(self.save_dir, "{}.h5".format(self.exp_name))
        return filename

    def read(self):
        # # For testing usage
        ret = {}
        file = self.file
        self.logger.debug(
            "Now we have everything in file: {}. self.dataset_names {}.".format(list(file.keys()), self.dataset_names))
        for k in self.dataset_names:
            dset = file[k]
            ret[k] = dset[:]
        return ret

    def close(self):
        if any([len(buffer) > 1 for buffer in self.buffers.values()]):
            length = len(self.buffers["timestamp"])
            for k, buffer in self.buffers.items():
                if len(buffer) != length:
                    self.logger.warning("The buffer have different length as timestamp! We will clip those excessive.")
                    buffer = buffer[:length]
                self._append_to_dataset(buffer, k)

        self.file.close()
        self.logger.debug('Recorder Disconnected. The whole life span of recorder is {} seconds.'.format(
            time.time() - self.created_timestamp))

def build_recorder_process(config, data_queue, log_queue, log_level):
    qh = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(qh)
    r = Recorder(config, logger)
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
        logger.info("Files has been saved at < {} >.".format(r.filename))
        r.close()
    logger.info("Prepare to delete data_queue and log_queue.")
    data_queue.cancel_join_thread()
    log_queue.cancel_join_thread()


if __name__ == '__main__':
    from utils import setup_logger
    import uuid
    log_level = "DEBUG"
    setup_logger(log_level)
    filename = "tmp_{}".format(uuid.uuid4())
    config = {"exp_name": None,
              "buffer_size": 5,
              "save_dir": 'tmp',
              "compress": False,
              "dataset_names": ("lidar_data", "extra_data", "frame", "timestamp"),
              "dataset_dtypes": {"lidar_data": "uint16", "extra_data": "float32", "frame": "uint8",
                                 "timestamp": "float32"},
              "dataset_shapes": {"lidar_data": (10, 100, 110), "extra_data": (10, 100, 110), "frame": (10, 100, 110),
                                 "timestamp": ()},
              }
    r = Recorder(config)
    for _ in range(103):
        data_dict = {k: np.empty([10, 100, 110]) for k in ("lidar_data", "extra_data", "frame")}
        r.add(data_dict)
    print(r.read())
    r.close()
