# =============================================================================
# Copyright (c) 2001-2018 FLIR Systems, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
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

def print_build_info():
    lib_ver = PyCapture2.getLibraryVersion()
    print('PyCapture2 library version: %d %d %d %d' % (lib_ver[0], lib_ver[1], lib_ver[2], lib_ver[3]))
    print()

def print_camera_info(cam):
    cam_info = cam.getCameraInfo()
    print('\n*** CAMERA INFORMATION ***\n')
    print('Serial number - %d', cam_info.serialNumber)
    print('Camera model - %s', cam_info.modelName)
    print('Camera vendor - %s', cam_info.vendorName)
    print('Sensor - %s', cam_info.sensorInfo)
    print('Resolution - %s', cam_info.sensorResolution)
    print('Firmware version - %s', cam_info.firmwareVersion)
    print('Firmware build time - %s', cam_info.firmwareBuildTime)
    print()

def save_video_helper(cam, file_format, filename, framerate):
    num_images = 100

    video = PyCapture2.FlyCapture2Video()

    for i in range(num_images):
        try:
            image = cam.retrieveBuffer()
        except PyCapture2.Fc2error as fc2Err:
            print('Error retrieving buffer : %s' % fc2Err)
            continue

        print('Grabbed image {}'.format(i))

        if (i == 0):
            if file_format == 'AVI':
                video.AVIOpen(filename, framerate)
            elif file_format == 'MJPG':
                video.MJPGOpen(filename, framerate, 75)
            elif file_format == 'H264':
                video.H264Open(filename, framerate, image.getCols(), image.getRows(), 1000000)
            else:
                print('Specified format is not available.')
                return

        video.append(image)
        print('Appended image %d...' % i)

    print('Appended {} images to {} file: {}...'.format(num_images, file_format, filename))
    video.close()

#
# Example Main
#

# Print PyCapture2 Library Information
print_build_info()

# Ensure sufficient cameras are found
bus = PyCapture2.BusManager()
num_cams = bus.getNumOfCameras()
print('Number of cameras detected: %d' % num_cams)
if not num_cams:
    print('Insufficient number of cameras. Exiting...')
    exit()

# Select camera on 0th index
cam = PyCapture2.Camera()
cam.connect(bus.getCameraFromIndex(0))

# Print camera details
print_camera_info(cam)

print('Starting capture...')
cam.startCapture()

print('Detecting frame rate from Camera')
fRateProp = cam.getProperty(PyCapture2.PROPERTY_TYPE.FRAME_RATE)
framerate = fRateProp.absValue

print('Using frame rate of {}'.format(framerate))

for file_format in ('AVI','H264','MJPG'):
    filename = 'SaveImageToAviEx_{}.avi'.format(file_format)
    save_video_helper(cam, file_format, filename.encode('utf-8'), framerate)

print('Stopping capture...')
cam.stopCapture()
cam.disconnect()

input('Done! Press Enter to exit...\n')