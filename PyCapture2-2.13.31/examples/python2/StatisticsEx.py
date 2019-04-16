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

def print_build_info():
    lib_ver = PyCapture2.getLibraryVersion()
    print 'PyCapture2 library version: %d %d %d %d' % (lib_ver[0], lib_ver[1], lib_ver[2], lib_ver[3])
    print

def print_camera_info(cam):
    cam_info = cam.getCameraInfo()
    print '\n*** CAMERA INFORMATION ***\n'
    print 'Serial number - %d' % cam_info.serialNumber
    print 'Camera model - %s' % cam_info.modelName
    print 'Camera vendor - %s' % cam_info.vendorName
    print 'Sensor - %s' % cam_info.sensorInfo
    print 'Resolution - %s' % cam_info.sensorResolution
    print 'Firmware version - %s' % cam_info.firmwareVersion
    print 'Firmware build time - %s' % cam_info.firmwareBuildTime
    print

def enable_embedded_timestamp(cam, enable_timestamp):
    embedded_info = cam.getEmbeddedImageInfo()
    if embedded_info.available.timestamp:
        cam.setEmbeddedImageInfo(timestamp = enable_timestamp)
        if enable_timestamp:
            print '\nTimeStamp is enabled.\n'
        else:
            print '\nTimeStamp is disabled.\n'

def grab_images(cam, num_images_to_grab):
    prev_ts = None
    image_list = []
    for i in xrange(num_images_to_grab):
        image = cam.retrieveBuffer()
        ts = image.getTimeStamp()
        if prev_ts:
            diff = (ts.cycleSeconds - prev_ts.cycleSeconds) * 8000 + (ts.cycleCount - prev_ts.cycleCount)
            print 'Timestamp [ %d %d %d ] - ' % (ts.cycleSeconds, ts.cycleCount, diff)
        prev_ts = ts

    print '\nPerforming statistics calculation...'
    testimg = image.convert(PyCapture2.PIXEL_FORMAT.MONO8)
    stats = PyCapture2.ImageStatistics()
    stats.enableGreyChannel()
    stats.calculateStatistics(testimg)
    channel_lookup = {'GREY': 0,'RED': 1, 'GREEN': 2, 'BLUE': 3, 'HUE': 4, 'SATURATION': 5, 'LIGHTNESS': 6}
    for key, value in channel_lookup.iteritems():
        if not stats.getChannelStatus(value):
            print '%s is not enabled.' % key
        else:
            print '%s:' % key
            max_range, val_range, num_vals, mean, histo = stats.getStatistics(value)
            print 'Range: {} to {}'.format(max_range[0], max_range[1])
            print 'Value Range: {} to {}'.format(val_range[0], val_range[1])
            print 'Number of Values: {}'.format(num_vals)
            print 'Mean value: {}'.format(mean)
            print 'Histogram: \n{}'.format(', '.join([str(hist) for hist in histo]))

#
# Example Main
#

# Print PyCapture2 Library Information
print_build_info()

# Ensure sufficient cameras are found
bus = PyCapture2.BusManager()
num_cams = bus.getNumOfCameras()
print 'Number of cameras detected: %d' % num_cams
if not num_cams:
    print 'Insufficient number of cameras. Exiting...'
    exit()

# Select camera on 0th index
c = PyCapture2.Camera()
uid = bus.getCameraFromIndex(0)
c.connect(uid)

# Print camera details
print_camera_info(c)

# Enable camera embedded timestamp
enable_embedded_timestamp(c, True)

print 'Starting image capture...'
c.startCapture()
grab_images(c, 10)
c.stopCapture()

# Disable camera embedded timestamp
enable_embedded_timestamp(c, False)
c.disconnect()

raw_input('Done! Press Enter to exit...\n')