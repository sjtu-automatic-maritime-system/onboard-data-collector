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

def print_build_info():
    lib_ver = PyCapture2.getLibraryVersion()
    print 'PyCapture2 library version: %d %d %d %d' % (lib_ver[0], lib_ver[1], lib_ver[2], lib_ver[3])
    print

def print_camera_info(cam_info):
    print '\n*** CAMERA INFORMATION ***\n'
    print 'Serial number - %d' % cam_info.serialNumber
    print 'Camera model - %s' % cam_info.modelName
    print 'Camera vendor - %s' % cam_info.vendorName
    print 'Sensor - %s' % cam_info.sensorInfo
    print 'Resolution - %s' % cam_info.sensorResolution
    print 'Firmware version - %s' % cam_info.firmwareVersion
    print 'Firmware build time - %s' % cam_info.firmwareBuildTime
    print
    print 'GigE major version - %d' % cam_info.gigEMajorVersion
    print 'GigE minor version - %d' % cam_info.gigEMinorVersion
    print 'User-defined name - %s' % cam_info.userDefinedName
    print 'XML URL1 - %s' % cam_info.xmlURL1
    print 'XML URL2 - %s' % cam_info.xmlURL2
    print 'MAC address - %d %d %d %d %d %d' % (cam_info.macAddress[0], cam_info.macAddress[1], cam_info.macAddress[2], cam_info.macAddress[3], cam_info.macAddress[4], cam_info.macAddress[5])
    print 'IP address - %d.%d.%d.%d' % (cam_info.ipAddress[0], cam_info.ipAddress[1], cam_info.ipAddress[2], cam_info.ipAddress[3])
    print 'Subnet mask - %d %d %d %d' % (cam_info.subnetMask[0], cam_info.subnetMask[1], cam_info.subnetMask[2], cam_info.subnetMask[3])
    print 'Default geteway - %d %d %d %d' % (cam_info.defaultGateway[0], cam_info.defaultGateway[1], cam_info.defaultGateway[2], cam_info.defaultGateway[3])
    print

def print_stream_channel_info(stream_info):
    print 'Network interface: %s' % stream_info.networkInterfaceIndex
    print 'Host port: %s' % stream_info.hostPort
    print 'Do not fragment bit: %s' % 'Enabled' if stream_info.doNotFragment else 'Disabled'
    print 'Packet size: %s' % stream_info.packetSize
    print 'Inter-packet delay: %s' % stream_info.interPacketDelay
    print 'Destination IP address: %d %d %d %d' % (stream_info.destinationIpAddress[0], stream_info.destinationIpAddress[1], stream_info.destinationIpAddress[2], stream_info.destinationIpAddress[3])
    print 'Source port (on Camera): %s' % stream_info.sourcePort
    print

def enable_embedded_timestamp(cam, enable_timestamp):
    embedded_info = cam.getEmbeddedImageInfo()
    if embedded_info.available.timestamp:
        cam.setEmbeddedImageInfo(timestamp = enable_timestamp)
        if enable_timestamp:
            print '\nTimeStamp is enabled.\n'
        else:
            print '\nTimeStamp is disabled.\n'

def run_single_camera(cam, uid):
    print 'Connecting to Camera...'
    cam.connect(uid)
    print_camera_info(cam.getCameraInfo())
    for i in range(cam.getNumStreamChannels()):
        print_stream_channel_info(cam.getGigEStreamChannelInfo(i))
    print 'Querying GigE image setting information...'
    img_settings_info = cam.getGigEImageSettingsInfo()
    image_settings = PyCapture2.GigEImageSettings()

    image_settings.offsetX = 0
    image_settings.offsetY = 0
    image_settings.height = img_settings_info.maxHeight
    image_settings.width = img_settings_info.maxWidth
    image_settings.pixelFormat = PyCapture2.PIXEL_FORMAT.MONO8

    print 'Setting GigE image settings...'
    cam.setGigEImageSettings(image_settings)
    enable_embedded_timestamp(cam, True)

    print 'Starting image capture...'
    cam.startCapture()
    prev_ts = None
    num_images_to_grab = 10
    for i in xrange(num_images_to_grab):
        try:
            image = cam.retrieveBuffer()
        except PyCapture2.Fc2error as fc2Err:
            print 'Error retrieving buffer : %d' % fc2Err
            continue

        ts = image.getTimeStamp()
        if prev_ts:
            diff = (ts.cycleSeconds - prev_ts.cycleSeconds) * 8000 + (ts.cycleCount - prev_ts.cycleCount)
            print 'Timestamp [ %d %d ] - %d' % (ts.cycleSeconds, ts.cycleCount, diff)
        prev_ts = ts

    newimg = image.convert(PyCapture2.PIXEL_FORMAT.BGR)
    print 'Saving the last image to GigEGrabEx.png'
    newimg.save('GigEGrabEx.png', PyCapture2.IMAGE_FILE_FORMAT.PNG)
    cam.stopCapture()
    enable_embedded_timestamp(cam, False)
    cam.disconnect()

#
# Example Main
#

# Print PyCapture2 Library Information
print_build_info()

# Ensure sufficient cameras are found
bus = PyCapture2.BusManager()
cam_infos = bus.discoverGigECameras()
for ci in cam_infos:
    print_camera_info(ci)

if not len(cam_infos):
    print 'No suitable GigE cameras found. Exiting...'
    exit()

# Run example on all cameras
cam = PyCapture2.GigECamera()
for i in range(bus.getNumOfCameras()):
    uid = bus.getCameraFromIndex(i)

    ifaceType = bus.getInterfaceTypeFromGuid(uid)
    if ifaceType == PyCapture2.INTERFACE_TYPE.GIGE:
        run_single_camera(cam, uid)

raw_input('Done! Press Enter to exit...\n')
