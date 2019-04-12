#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import *
from socket import *
from struct import *
from msgdev import MsgDevice,PeriodTimer
import numpy as np

from config import VLPConfig



import time




vlp_config = VLPConfig()

# User Settings
VLP16_Connected = 1 #if not connected, this program will use self-created lidar data
data_addr = 'tcp://0.0.0.0:{}'.format(vlp_config.vlp_port) #send data to this address, including dis384,ref384,azi24
# data_addr = 'tcp://192.168.1.222' #send data to this address, including dis384,ref384,azi24
t_refresh = 1 #period to refresh

#VLP16 Settings
vlp_frequency = 10 #frequency of vlp16, matching its setting on web
vlp_returnmode = b'\x37' #37-Strongest Return,38-Last Return. Duel Return mode is not available here
max_ref = 255 #a point with reflectivity larger than this value will be regarded as noise, up to 255
min_ref = 0  #a point with reflectivity lower than this value will be regarded as noise, down to 0

# Parameter Calculation
# vlp_addr = ('192.168.1.255',2368) #ip and port of vlp16
# vlp_addr = (INADDR_ANY,2368) #ip and port of vlp16
res_azi = vlp_frequency*2  #resolution of azimuth angle
num_packet = ceil(36000/(res_azi*24)) #number of packets in one image, 150 when vlp_frequency is 5Hz.
packet_created = (b'\xff\xee\x33\x71'+b'\x89\x59\x17'*32)*12+b'\x61\x67\xb9\x5a\x37\x22' #when vlp16 is not connected, use this packet to debug

# Structure of vlp16 raw data
PacketTail = vlp_returnmode+b'\x22'
BlockHeader = b'\xff\xee'
azi_struct = Struct("<H")
dis_struct = Struct("<HB")
time_struct = Struct("<L")
factory_struct = Struct("<2B")

# Structure of vlp16 packet
flag = np.dtype('<u2')
azimuth  = np.dtype('<u2')
distance = np.dtype('<u2')
reflectivity = np.dtype('<u1')
channel = np.dtype([('distance',distance,1),('reflectivity',reflectivity,1)])
block = np.dtype([('flag',flag,1),('azimuth',azimuth,1),('channel',channel,32)])
packet = np.dtype([('block',block,12)])


def devinitial(dev):
    dev.pub_bind(data_addr)
    dev.sub_connect('tcp://{}:55004'.format(vlp_config.local_ip))
    dev.sub_connect('tcp://{}:55005'.format(vlp_config.local_ip))
    dev.sub_connect('tcp://{}:55204'.format(vlp_config.local_ip))
    dev.sub_add_url('ahrs.roll')
    dev.sub_add_url('ahrs.pitch')
    dev.sub_add_url('ahrs.yaw')
    dev.sub_add_url('gps.posx')
    dev.sub_add_url('gps.posy')
    dev.sub_add_url('gps.runtime')
    dev.sub_add_url('gps102.posx')
    dev.sub_add_url('gps102.posy')


class VLP:
    def __init__(self,dev):
        #create socket for UDP client
        try:
            self.s = socket(AF_INET,SOCK_DGRAM)
        except socket.error as msg:
            print('Failed to create socket. Error code:' + str(msg[0]) + ', Error message:' + msg[1])
            raise
        else:
            print('Socket Created.')

        #connect UDP client to server
        try:
            # self.s.connect(vlp_addr)
            self.s.bind(('',vlp_config.vlp_raw_port))
        except Exception:
            print('Failed to connect.')
            raise
        else:
            print('VLP-16 Connected.')
        self.s.settimeout(2)
        self.dev = dev
        self.runtime = 0

    def capture(self):
        if VLP16_Connected != 0:
            self.buf = self.s.recv(1206)   #length of a packet is 1206
            # self.buf = self.s.recv(1248)   #length of a packet is 1206
            # confirm the packet is ended with PacketTail
            while self.buf[1204:1206] != PacketTail:
                print('Wrong Factory Information!')
                self.buf = self.s.recv(1206)
        else:
            self.buf = packet_created

    def parse(self):
        datas = np.frombuffer(self.buf,dtype=packet,count=1)
        self.dis384 = datas['block']['channel']['distance']
        self.ref384 = datas['block']['channel']['reflectivity']
        azi12ori = datas['block']['azimuth']
        azi12ori = azi12ori.reshape(12,1)
        azi12add = azi12ori + res_azi
        azi12add = azi12add-(azi12add>=36000)*36000
        self.azi24 = np.column_stack((azi12ori,azi12add))

    def publish(self):
        if vlp_config.fake_run_time:
            self.runtime += 1
        else:
            self.runtime = self.dev.sub_get1('gps.runtime')
        posx = self.dev.sub_get1('gps.posx')
        posy = self.dev.sub_get1('gps.posy')
        posx102 = self.dev.sub_get1('gps102.posx')
        posy102 = self.dev.sub_get1('gps102.posy')
        roll = dev.sub_get1('ahrs.roll')
        pitch = dev.sub_get1('ahrs.pitch')
        yaw = dev.sub_get1('ahrs.yaw')
        # self.image[-8:] = [posx,posy,roll,pitch,yaw,posx102,posy102,self.runtime]
        self.image += [posx,posy,roll,pitch,yaw,posx102,posy102,self.runtime]
        self.dev.pub_set('vlp.image', self.image)

    def update(self):
        # self.image = np.empty((num_packet * 408 + 8), dtype=np.float32)
        self.image = []
        for i in range(num_packet):
            self.capture()
            self.parse()
            self.packet = np.hstack((self.dis384.flatten(),self.azi24.flatten()))
            self.image = self.image+self.packet.tolist()
            # self.image[408 * i: 408 * (i+1)] = self.packet
        self.publish()


    def close(self):
        self.s.close()
        print('VLP-16 Disconnected.')




from utils import get_formatted_time
if __name__ == "__main__":
    try:
        dev = MsgDevice()
        dev.open()
        devinitial(dev)
        vlp = VLP(dev)
        t = PeriodTimer(t_refresh)
        t.start()
        while True:
            with t:
                vlp.update()
                # print(vlp.image)
                print("[INFO] {}: {}".format(get_formatted_time(), "raw lidar data received! Shape {}, max {}, min {}.".format(len(vlp.image), max(vlp.image), min(vlp.image))))
    except KeyboardInterrupt or Exception:
        vlp.close()
        dev.close()
        raise
    finally:
        vlp.close()
        dev.close()
