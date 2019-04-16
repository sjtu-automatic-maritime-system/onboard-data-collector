# =============================================================================
# Copyright 2018 FLIR Integrated Imaging Solutions, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ('Confidential Information'). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
# =============================================================================

import PyCapture2
from time import sleep
from sys import exit

num_images = 0

def print_build_info():
    libVer = PyCapture2.getLibraryVersion()
    print('FlyCapture2 library version: %d %d %d %d' % (libVer[0], libVer[1], libVer[2], libVer[3]))

def print_camera_info(cam):
    cam_info = cam.getCameraInfo()
    print('\n*** CAMERA INFORMATION ***\n')
    print('Serial number - %d' % cam_info.serialNumber)
    print('Camera model - %s' % cam_info.modelName)
    print('Camera vendor - %s' % cam_info.vendorName)
    print('Sensor - %s' % cam_info.sensorInfo)
    print('Resolution - %s' % cam_info.sensorResolution)
    print('Firmware version - %s' % cam_info.firmwareVersion)
    print('Firmware build time - %s' % cam_info.firmwareBuildTime)

def on_image_grabbed(img, val):
    global num_images
    num_images += 1
    print('Grabbed image %d' % num_images)

def grab_images(cam, num_images_to_grab):
    c.startCapture(on_image_grabbed, 3)
    print('Getting framerate!')
    framerate_property = cam.getProperty(PyCapture2.PROPERTY_TYPE.FRAME_RATE)
    framerate = framerate_property.absValue
    
    while num_images < num_images_to_grab:
        print('Start sleep!')
        sleep(1.0/framerate)
        print('End sleep!')

#
# Example Main
#

# Print PyCapture2 Library Information
print_build_info()

# Ensure sufficient cameras are found
bus = PyCapture2.BusManager()
c = PyCapture2.Camera()
if not bus.getNumOfCameras():
    print('No cameras detected')
    exit()

# Run example on the first camera
uid = bus.getCameraFromIndex(0)
c.connect(uid)
print_camera_info(c)
grab_images(c, 10)
c.stopCapture()

input('Done! Press Enter to exit...')