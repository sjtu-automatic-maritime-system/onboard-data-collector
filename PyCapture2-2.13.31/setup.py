"""
compile & install FlyCap2 for windows and linux.

Directory should appear something like:

PyCapture2
|-doc
| |-FlyCap2 documentation.chm
| +-FlyCap2 documentation.pdf
|
|-src
| |-python2
| | +-PyCapture2.c
| |
| |-python3
| | +-PyCapture2.c
|
|-examples
| |-python2
| | + <python 2 examples>
| +-python3
|   + <python 2 examples>
|
|-setup.py
|-README.txt
|-README_Linux.txt
|-README_MacOS.txt

"""
import os
import platform
import numpy as np
try:
    from setuptools import setup
    from setuptools.extension import Extension
except ImportError:
    from distutils import setup
    from distutils.extension import Extension
import struct
import sys

def getReadme():
    if os.name == 'posix':
        if platform.system() == 'Darwin':
            filedata = open("README_MacOS.txt")
        else:
            filedata = open("README_Linux.txt")
    else:
        filedata = open("README.txt")

    desc = filedata.read()
    filedata.close()
    return desc

os.chdir(sys.path[0])   #Change working directory to the script's directory

#if operating system is linux:
if os.name == 'posix':
    libDir = r'..\..\lib'
    incDir = r'..\..\include\C'
    libName = 'flycapture-c'
    libVideoName = 'flycapturevideo-c'

#if operating system is windows:
else:
    #winreg has different names in python 2 & 3
    if sys.version_info[0] < 3:
        import _winreg as winreg
    else:
        import winreg

    #check if 32 or 64bit python - installed flycapture 2 must match!
    if struct.calcsize("P") == 8:   #64bit
        libDir = os.path.abspath(r'..\..\lib64\C')
    else:                           #not 64bit - 32bit
        libDir = os.path.abspath(r'..\..\lib\C')

    incDir = os.path.abspath(r'..\..\include\C')
    libName = 'FlyCapture2_C'
    libVideoName = 'FlyCapture2Video_C'

#python 2 and python 3 are different - each must be cythonized differently.
if sys.version_info[0] < 3:
    srcDir = os.path.normcase(r'src/python2/')
else:
    srcDir = os.path.normcase(r'src/python3/')


#put include path, library, etc into extension
extensions = [
    Extension('PyCapture2', [srcDir + r'PyCapture2.c'],
        include_dirs = [incDir, np.get_include()],
        library_dirs = [libDir],
        libraries = [libName, libVideoName]
    )
]

#specify docs, create list of all additional files to add.
eventsTable = ('', ['pgr_events_table.dat'])
dataFiles = [eventsTable]

#get version from arguments (if it exists)
if '--setVer' in sys.argv:
    verInd = sys.argv.index('--setVer')
    sys.argv.remove('--setVer')
    ver = sys.argv[verInd]
    sys.argv.remove(ver)
else:
    ver = ''

setup(
    name = "PyCapture2",
    version = ver,
    author = "FLIR Integrated Imaging Solutions, Inc",
    description = "A python wrapper for the FlyCapture 2 library.",
    url = "https://www.ptgrey.com/",
    long_description = getReadme(),
    data_files = dataFiles,
    ext_modules = extensions
)
