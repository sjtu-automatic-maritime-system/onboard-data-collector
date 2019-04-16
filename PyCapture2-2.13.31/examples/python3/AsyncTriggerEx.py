#=============================================================================
# Copyright 2017 FLIR Integrated Imaging Solutions, Inc. All Rights Reserved.
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
#=============================================================================

import PyCapture2
from sys import exit
from time import sleep

def print_build_info():
    lib_ver = PyCapture2.getLibraryVersion()
    print('PyCapture2 library version: %d %d %d %d' % (lib_ver[0], lib_ver[1], lib_ver[2], lib_ver[3]))
    print()

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
    print()

def check_software_trigger_presence(cam):
    trigger_inq = 0x530
    if(cam.readRegister(trigger_inq) & 0x10000 != 0x10000):
        return False
    return True

def poll_for_trigger_ready(cam):
    software_trigger = 0x62C
    while True:
        reg_val = cam.readRegister(software_trigger)
        if not reg_val:
            break

def fire_software_trigger(cam):
    software_trigger = 0x62C
    fire_val = 0x80000000
    cam.writeRegister(software_trigger, fire_val)

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

c = PyCapture2.Camera()
c.connect(bus.getCameraFromIndex(0))

# Power on the Camera
CAMERA_POWER = 0x610
POWER_VAL = 0x80000000

c.writeRegister(CAMERA_POWER, POWER_VAL)

# Waiting for Camera to power up
retries = 10
time_to_sleep = 0.1 # seconds
for i in range(retries):
    sleep(time_to_sleep)
    try:
        reg_val = c.readRegister(CAMERA_POWER)
    except PyCapture2.Fc2error: # Camera might not respond to register reads during powerup.
        pass
    awake = True
    if reg_val == POWER_VAL:
        break
    awake = False
if not awake:
    print('Could not wake Camera. Exiting...')
    exit()

# Print camera details
print_camera_info(c)

# Configure trigger mode
trigger_mode = c.getTriggerMode()
trigger_mode.onOff = True
trigger_mode.mode = 0
trigger_mode.parameter = 0
trigger_mode.source = 7     # Using software trigger

c.setTriggerMode(trigger_mode)

poll_for_trigger_ready(c)

c.setConfiguration(grabTimeout = 5000)

# Start acquisition
c.startCapture()

if not check_software_trigger_presence(c):
    print('SOFT_ASYNC_TRIGGER not implemented on this Camera! Stopping application')
    exit()

# Grab images
num_images = 10
for i in range(num_images):
    poll_for_trigger_ready(c)
    print('\nPress the Enter key to initiate a software trigger')
    input()
    fire_software_trigger(c)
    try:
        image = c.retrieveBuffer()
    except PyCapture2.Fc2error as fc2Err:
        print('Error retrieving buffer : %s' % fc2Err)
        continue

    print('.')

c.setTriggerMode(onOff = False)
print('Finished grabbing images!')

c.stopCapture()
c.disconnect()

input('Done! Press Enter to exit...\n')
