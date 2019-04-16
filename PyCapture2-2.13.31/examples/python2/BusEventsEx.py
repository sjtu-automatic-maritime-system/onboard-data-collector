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
from time import time

def print_build_info():
    lib_ver = PyCapture2.getLibraryVersion()
    print 'PyCapture2 library version: %d %d %d %d' % (lib_ver[0], lib_ver[1], lib_ver[2], lib_ver[3])
    print

def on_bus_reset(serialNum):
    print '{} - *** BUS RESET ***'.format(time())

def on_bus_arrival(serialNum):
    print '{} - *** BUS ARRIVAL ***'.format(time())

def on_bus_removal(serialNum):
    print '{} - *** BUS REMOVAL ***'.format(time())

#
# Example Main
#

# Print PyCapture2 Library Information
print_build_info()

# Register bus arrival/removal/reset callbacks
bus = PyCapture2.BusManager()
bus.registerCallback(PyCapture2.BUS_CALLBACK_TYPE.BUS_RESET, on_bus_reset)
bus.registerCallback(PyCapture2.BUS_CALLBACK_TYPE.ARRIVAL, on_bus_arrival)
bus.registerCallback(PyCapture2.BUS_CALLBACK_TYPE.REMOVAL, on_bus_removal)

raw_input('Done! Press Enter to exit...\n')

# Un-register bus arrival/removal/reset callbacks
bus.unregisterCallback(PyCapture2.BUS_CALLBACK_TYPE.BUS_RESET)
bus.unregisterCallback(PyCapture2.BUS_CALLBACK_TYPE.ARRIVAL)
bus.unregisterCallback(PyCapture2.BUS_CALLBACK_TYPE.REMOVAL)
