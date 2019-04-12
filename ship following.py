#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
import time
from math import *

import numpy as np

from msgdev import MsgDevice, PeriodTimer

# record setting
# DocumentAddress = 'C:/Users/ZhangLei/Desktop/0919'
import os
import os.path as osp
DocumentAddress = 'test'
if not osp.exists(DocumentAddress):
    os.mkdir(DocumentAddress)



# global map parameter
num_out_x = 200
num_out_y = 200

# period
t_refresh = 0.5

# pid control parameter
kp = 900
kd = 1300

# navigation parameter
rpm_normal_left = [800, 1200, 1400]
rpm_normal_right = [850, 1250, 1450]

# rpm limit
rpm_max = 2000


def devinitial(dev):
    dev.open()
    dev.pub_bind('tcp://0.0.0.0:55002')
    # gps
    dev.sub_connect('tcp://192.168.1.150:55004')
    dev.sub_add_url('gps.posx')
    dev.sub_add_url('gps.posy')
    dev.sub_add_url('gps.runtime')
    # gps102
    dev.sub_connect('tcp://192.168.1.150:55204')
    dev.sub_add_url('gps102.posx')
    dev.sub_add_url('gps102.posy')
    # ahrs
    dev.sub_connect('tcp://192.168.1.150:55005')
    dev.sub_add_url('ahrs.yaw')
    dev.sub_add_url('ahrs.yaw_speed')
    # map and objects
    dev.sub_connect('tcp://127.0.0.1:55012')
    dev.sub_add_url('map.objnum')
    dev.sub_add_url('map.globalmap', [0, ] * (num_out_x * num_out_y))
    dev.sub_add_url('map.objects', [0, ] * 1700)
    dev.sub_add_url('map.pos&time', [0, ] * 3)


def diff_adjust(diff):
    while diff > pi:
        diff -= 2 * pi
    while diff < -pi:
        diff += 2 * pi
    return diff


def pid(diff, lastdiff):
    return diff * kp + (diff - lastdiff) / t_refresh * kd


def rpm_limit(rpm):
    if rpm > rpm_max:
        return rpm_max
    elif rpm < -rpm_max:
        return -rpm_max
    else:
        return rpm


if __name__ == "__main__":
    dev = MsgDevice()
    sf_doc = DocumentAddress + '/shipfollowing_' + time.strftime('%Y-%m-%d-%H-%M-%S',
                                                                 time.localtime()) + '.csv'  # name and address of image data
    with open(sf_doc, 'w', newline='') as sf_csvfile:
        sf_writer = csv.writer(sf_csvfile)

    try:
        devinitial(dev)
        inf = np.zeros((1, 17))
        lastdiff = 0
        teststart = 0
        shiplabel = -1  # which object is followed
        t = PeriodTimer(t_refresh)
        t.start()
        while True:
            with t:
                objnum = int(ceil(dev.sub_get1('map.objnum')))
                if objnum > 0 and teststart == 0:
                    teststart = 1
                    print('Test starts now!')
                objects = np.fromiter(dev.sub_get('map.objects'), 'd').reshape(100, 17)
                print('Max confidence level:', objects[:, 14].max())
                mapinf = np.fromiter(dev.sub_get('map.pos&time'), 'd').reshape(1, 3)
                mapposx = mapinf[0, 0]
                mapposy = mapinf[0, 1]
                maptime = mapinf[0, 2]
                posx = dev.sub_get1('gps.posx')
                posy = dev.sub_get1('gps.posy')
                gpstime = dev.sub_get1('gps.runtime')
                print('Map time:', maptime, 'GPS time:', gpstime)
                posx102 = dev.sub_get1('gps102.posx')
                posy102 = dev.sub_get1('gps102.posy')
                heading = dev.sub_get1('ahrs.yaw')
                ship = objects[np.where(objects[:, 16] == shiplabel)]
                if len(ship) == 0:
                    for i in range(objnum):
                        object_i = objects[i]
                        if object_i[14] >= 12 and object_i[5] < 1.5:
                            shiplabel = object_i[16]
                            ship = object_i.reshape(1, 17)
                            break
                if len(ship) != 1:
                    print('Sorry, no ship to follow.')
                    print('Target GPS', posx102, posy102)
                    lastdiff = 0
                    left = 0
                    right = 0
                else:
                    print('Going to follow No.%d object!' % (shiplabel), 'Confidence level:', ship[0, 14])
                    ship_x = (ship[0, 1] + ship[0, 2]) / 2
                    ship_y = (ship[0, 3] + ship[0, 4]) / 2
                    dis = sqrt((ship_x - posx) ** 2 + (ship_y - posy) ** 2)
                    heading_goal = atan2(ship_y - posy, ship_x - posx)
                    diff = diff_adjust(heading_goal - heading)
                    print('Goalheading-Heading:', diff)
                    diff_rpm = pid(diff, lastdiff)
                    print('Leftrpm-Rightrpm=', diff_rpm)
                    lastdiff = diff
                    if dis <= 3:
                        left = rpm_normal_left[0] + diff_rpm
                        right = rpm_normal_right[0] - diff_rpm
                    elif dis <= 10:
                        left = rpm_normal_left[1] + diff_rpm
                        right = rpm_normal_right[1] - diff_rpm
                    else:
                        left = rpm_normal_left[2] + diff_rpm
                        right = rpm_normal_right[2] - diff_rpm
                    print('Location:', mapposx, mapposy, 'Location GPS:', posx, posy, 'Target:', ship_x, ship_y,
                          'Target GPS', posx102, posy102, 'Distance:', dis, 'Goal heading:', heading_goal, 'Heading:',
                          heading)
                left = rpm_limit(left)
                right = rpm_limit(right)
                left = -left
                dev.pub_set1('pro.left.speed', left)
                dev.pub_set1('pro.right.speed', right)
                print('Motor:', left, right)
                inf[0, 0:9] = np.array([objnum, posx, posy, posx102, posy102, heading, left, right, shiplabel])
                if teststart == 1:
                    sf_writer.writerows(inf)
                    sf_writer.writerows(objects)

    except (KeyboardInterrupt, Exception) as e:
        dev.pub_set1('pro.left.speed', 0)
        dev.pub_set1('pro.right.speed', 0)
        dev.close()
        sf_csvfile.close()
        raise
