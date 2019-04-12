#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
from math import cos, sin, ceil, pi, floor

import numpy as np

from config import ImageConfig
from msgdev import MsgDevice, PeriodTimer


# VLP16 Settings
vlp_frequency = 10  # frequency of vlp16, matching its setting on web

# Image Settings
image_size = 50  # side length of a square image
grid_size = 0.2  # grid size in image
min_dis = 1  # ignore reflection nearer than the minimum distance
vrange = 1.5  # vertical value range from -vrange to vrange

# Parameter Calculation
res_azi = vlp_frequency * 2  # resolution of azimuth angle
num_packet = ceil(36000 / (res_azi * 24))  # number of packets in one image
num_grid = ceil(image_size / grid_size)
vang_mat = np.zeros((8, 2))  # vertical angle matrix(8*2)
for p in range(8):
    for q in range(2):
        vang_mat[p, q] = (q * 16 + p * 2 - 15) / 180 * pi


image_config = ImageConfig()


def devinitial(dev):
    dev.pub_bind('tcp://0.0.0.0:{}'.format(image_config.image_port))
    dev.sub_connect('tcp://{}:{}'.format(image_config.local_ip, image_config.vlp_port))
    # dev.sub_connect('tcp://{}:{}'.format("127.0.0.1", image_config.vlp_port))


from utils import get_formatted_time


class Image:
    def __init__(self, dev):
        # create data document
        self.rot_m = np.zeros((3, 3))  # rotation matrix
        self.dev = dev
        self.dev.sub_add_url('vlp.image', [0, ] * (408 * num_packet + 8))
        self.runtime = 0
        self.posx = 0
        self.posy = 0
        self.image_ini()
        if image_config.if_record_rawdata:
            self.raw_doc = image_config.log_dir + '/Raw_VLP16_' + get_formatted_time() + '.csv'  # name and address of image data
            self.raw_csvfile = open(self.raw_doc, 'w', newline='')
            self.raw_writer = csv.writer(self.raw_csvfile)

    def image_ini(self):
        self.image_high = np.ones([num_grid, num_grid]) * (-vrange)
        self.image_low = np.ones([num_grid, num_grid]) * (vrange)
        self.image_height = np.zeros([num_grid, num_grid])
        self.image_n = np.zeros([num_grid, num_grid])
        self.image_refmean = np.zeros([num_grid, num_grid])

    def sub_from_vlp(self):
        self.image = self.dev.sub_get('vlp.image')

    def rotmatrix(self, roll, pitch, yaw):
        self.rot_m[0, 0] = cos(pitch) * cos(yaw)
        self.rot_m[0, 1] = sin(roll) * sin(pitch) * cos(yaw) - cos(roll) * sin(yaw)
        self.rot_m[0, 2] = cos(roll) * sin(pitch) * cos(yaw) + sin(roll) * sin(yaw)
        self.rot_m[1, 0] = cos(pitch) * sin(yaw)
        self.rot_m[1, 1] = sin(roll) * sin(pitch) * sin(yaw) + cos(roll) * cos(yaw)
        self.rot_m[1, 2] = cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw)
        self.rot_m[2, 0] = -sin(pitch)
        self.rot_m[2, 1] = sin(roll) * cos(pitch)
        self.rot_m[2, 2] = cos(roll) * cos(pitch)

    def parse(self):
        if image_config.if_record_rawdata:
            self.raw_writer.writerows(np.array(
                [self.posx, self.posy, self.roll, self.pitch, self.yaw, self.posx102, self.posy102,
                 self.runtime]).reshape(1, 8))
        datas = np.fromiter(self.image, 'd')
        for k in range(num_packet):
            dis384 = datas[k * 408:(k * 408 + 384)].reshape(12, 2, 8, 2)
            azi24 = datas[(k * 408 + 384):(k * 408 + 408)].reshape(12, 2) / 18000 * pi
            for i in range(12):
                for j in range(2):
                    for p in range(8):
                        for q in range(2):
                            dis = dis384[i, j, p, q] * 0.002
                            vang = (q * 16 + p * 2 - 15) / 180 * pi
                            azi = azi24[i, j]
                            if image_config.if_record_rawdata:
                                self.raw_writer.writerows(np.array([dis, azi, vang, 0, 0, 0, 0, 0]).reshape(1, 8))
                            if dis > min_dis:
                                z = -dis * sin(vang)
                                x = dis * cos(vang) * cos(azi)
                                y = dis * cos(vang) * sin(azi)
                                v_move = np.zeros(3)
                                v_move[0], v_move[1], v_move[2] = x, y, z
                                v_fix = self.rot_m.dot(v_move)
                                x, y, z = v_fix[0], v_fix[1], v_fix[2]
                                self.drawpoint(x, y, -z)

    def drawpoint(self, x, y, h):
        if h < vrange and h > -vrange:
            pos_x = floor((x + image_size / 2) / grid_size)
            pos_y = floor((y + image_size / 2) / grid_size)
            try:
                if pos_x < num_grid and pos_x >= 0 and pos_y >= 0 and pos_y < num_grid:
                    high = self.image_high[pos_x, pos_y]
                    low = self.image_low[pos_x, pos_y]
                    if h > high:
                        self.image_high[pos_x, pos_y] = h
                    if h < low:
                        self.image_low[pos_x, pos_y] = h
                    self.image_n[pos_x, pos_y] = self.image_n[pos_x, pos_y] + 1
            except Exception:
                pass
        else:
            pass

    def publish(self):
        self.dev.pub_set1('image.posx', self.posx)
        self.dev.pub_set1('image.posy', self.posy)
        self.dev.pub_set1('image.runtime', self.runtime)
        self.dev.pub_set('image.image_high', self.image_high.flatten().tolist())
        self.dev.pub_set('image.image_low', self.image_low.flatten().tolist())
        self.dev.pub_set('image.image_n', self.image_n.flatten().tolist())

    def update(self):
        self.image_ini()
        while True:
            self.sub_from_vlp()
            if self.image[-1] != self.runtime:
                print("[DEBUG] runtime updated! now is ", self.runtime)
                print("[DEBUG] {}:self.image shape {}, self.image max {}, self.image min {}.".format(get_formatted_time(),
                                                                                                     len(self.image),
                                                                                                  max(self.image),
                                                                                                  min(self.image)))
                self.runtime = self.image[-1]
                break
        [self.posx, self.posy, self.roll, self.pitch, self.yaw, self.posx102, self.posy102] = self.image[-8:-1]
        self.rotmatrix(self.roll, self.pitch, self.yaw)
        print("Enter parse")
        self.parse()
        print("Enter publish")
        self.publish()

    def close(self):
        if image_config.if_record_rawdata:
            self.raw_csvfile.close()
        self.dev.close()
        print('VLP-16 Image Disconnected.')


if __name__ == "__main__":
    try:
        dev = MsgDevice()
        dev.open()
        devinitial(dev)
        myimage = Image(dev)
        t = PeriodTimer(image_config.image_refresh_interval)
        t.start()
        print("Enter loop")
        while True:
            with t:
                print("Prepare to update at", myimage.runtime)
                myimage.update()
                print('Run Time Image:', myimage.runtime)

    except KeyboardInterrupt:
        myimage.close()
    except Exception:
        myimage.close()
        raise
    else:
        myimage.close()
    finally:
        pass
