import PyCapture2
import logging
import numpy as np

def setup_camera():
    bus = PyCapture2.BusManager()
    num_cams = bus.getNumOfCameras()
    logging.info('Number of cameras detected: %d' % num_cams)
    if not num_cams:
        logging.error('Insufficient number of cameras. Exiting...')
        raise ValueError('Insufficient number of cameras. Exiting...')

    cam = PyCapture2.Camera()
    cam.connect(bus.getCameraFromIndex(0))
    cam_info = cam.getCameraInfo()
    logging.info('*** CAMERA INFORMATION ***')
    logging.info('Serial number - %d', cam_info.serialNumber)
    logging.info('Camera model - %s', cam_info.modelName)
    logging.info('Camera vendor - %s', cam_info.vendorName)
    logging.info('Sensor - %s', cam_info.sensorInfo)
    logging.info('Resolution - %s', cam_info.sensorResolution)
    logging.info('Firmware version - %s', cam_info.firmwareVersion)
    logging.info('Firmware build time - %s', cam_info.firmwareBuildTime)
    cam.startCapture()
    return cam

def shot(cam):
    image = cam.retrieveBuffer()
    image = image.convert(PyCapture2.PIXEL_FORMAT.BGR)
    image2 = np.array(image.getData(), dtype=np.uint8).reshape(image.getRows(), image.getCols(), 3)
    return image2

def close_camera(cam):
    cam.stopCapture()
    cam.disconnect()
    logging.info('Camara Disconnected.')

if __name__ == '__main__':
    import cv2
    from utils import setup_logger
    setup_logger('DEBUG')

    cam = setup_camera()
    try:
        while True:
            img = shot(cam)
            cv2.imshow('test', img)
            cv2.waitKey(1)
    finally:
        cv2.destroyAllWindows()
        close_camera(cam)