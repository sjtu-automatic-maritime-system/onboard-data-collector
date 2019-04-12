#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import*
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from msgdev import MsgDevice,PeriodTimer
import time
import csv
import os
import os.path as osp
# DocumentAddress = 'C:/Users/ZhangLei/Desktop/0919'
DocumentAddress = 'test'
if not osp.exists(DocumentAddress):
    os.mkdir(DocumentAddress)

doctime = time.localtime()
grid_size = 0.2
num_out_x = 200
num_out_y = 200
num_grid = 250
vlp_frequency = 10
res_vlp = vlp_frequency*2/100/57.3
minlevel_map = 10
drawpicture = 1

#Standard number image settings
image_n_std = np.zeros((num_grid,num_grid))
for p in range(num_grid):
    for q in range(num_grid):
        if p!=(num_grid/2) or q!=(num_grid/2):
            dis_pq = grid_size*sqrt((p-num_grid/2)**2+(q-num_grid/2)**2)
            n_std_pq = ceil(grid_size/(dis_pq*res_vlp))
            image_n_std[p,q] = n_std_pq*0.2

def sub_image(subnum,data,picsize,title='',xlabel='',ylabel='',cmaptype='viridis'):#cmaptype='tab20'
    plt.subplot(subnum)
    img  = data*1
    for i in range(picsize):
        img[:,i] = img[:,i][::-1]
    imgplot = plt.imshow(img,cmap=cmaptype)
    if xlabel != '':
        plt.xlabel(xlabel)
    if ylabel != '':
        plt.ylabel(ylabel)
    plt.colorbar()
    if title != '':
        plt.title(title)


if __name__ == "__main__":
    try:
        dev = MsgDevice()
        dev.open()
        dev.sub_connect('tcp://127.0.0.1:55012')
        dev.sub_add_url('map.objnum')
        dev.sub_add_url('map.globalmap_size',[0,]*6)
        dev.sub_add_url('map.timestamp')
        dev.sub_add_url('map.globalmap',[0,]*(num_out_x*num_out_y))
        dev.sub_add_url('map.objects',[0,]*1700)
        dev.sub_add_url('map.image_high',[0,]*(num_grid*num_grid))
        dev.sub_add_url('map.image_low',[0,]*(num_grid*num_grid))
        dev.sub_add_url('map.image_n',[0,]*(num_grid*num_grid))
        dev.sub_add_url('map.image_cluster',[0,]*(num_grid*num_grid))
        dev.sub_add_url('map.image_gain',[0,]*(num_grid*num_grid))
        dev.sub_add_url('map.pos&time',[0,]*3)
        dev.sub_add_url('image.posx102')
        dev.sub_add_url('image.posy102')

        #image document
        image_doc = DocumentAddress+'/Image_VLP16_'+time.strftime('%Y-%m-%d-%H-%M-%S',doctime)+'.csv'  #name and address of image data
        image_csvfile = open(image_doc,'w',newline='')
        image_writer = csv.writer(image_csvfile)

        #object document
        obj_doc = DocumentAddress+'/Objects_VLP16_'+time.strftime('%Y-%m-%d-%H-%M-%S',doctime)+'.csv'  #name and address of image data
        obj_csvfile = open(obj_doc,'w',newline='')
        obj_writer = csv.writer(obj_csvfile)

        inf  = np.zeros((1,num_grid))
        oldtime = 0
        while True:
            newtime = dev.sub_get1('map.timestamp')
            print(newtime)
            if newtime != oldtime:
                print('No.%d'%newtime)
                oldtime = newtime
                objnum = dev.sub_get1('map.objnum')*np.ones((1,17))
                print('Number of objects:',objnum[0,0])
                image_high = np.fromiter(dev.sub_get('map.image_high'),'d').reshape(num_grid,num_grid)
                image_low = np.fromiter(dev.sub_get('map.image_low'),'d').reshape(num_grid,num_grid)
                image_n = np.fromiter(dev.sub_get('map.image_n'),'d').reshape(num_grid,num_grid)
                image_cluster = np.fromiter(dev.sub_get('map.image_cluster'),'d').reshape(num_grid,num_grid)
                image_gain = np.fromiter(dev.sub_get('map.image_gain'),'d').reshape(num_grid,num_grid)
                globalmap = np.fromiter(dev.sub_get('map.globalmap'),'d').reshape(num_out_x,num_out_y)
                globalmap_level = np.zeros((num_out_x,num_out_y))
                objects = np.fromiter(dev.sub_get('map.objects'),'d').reshape(100,17)
                inf[0,0:3] = np.fromiter(dev.sub_get('map.pos&time'),'d').reshape(1,3) 
                posx = inf[0,0]
                posy = inf[0,1]
                posx102 = dev.sub_get1('image.posx102')
                posy102 = dev.sub_get1('image.posy102')
                objnum[0,-1] = posy102
                objnum[0,-2] = posx102

                #record data
                image_writer.writerows(inf)
                image_writer.writerows(image_high)
                image_writer.writerows(image_low)
                image_writer.writerows(image_n)
                obj_writer.writerows(objnum)
                obj_writer.writerows(objects)

                #draw and save global map
                if drawpicture != 0 and ceil(newtime)%10 == 1:
                    image_n[np.where(image_n>20)] = 20
                    image_cluster[np.where(image_cluster==0)] = -5
                    plt.figure(figsize=(5,5))
                    sub_image(111,image_high,num_grid)
                    plt.savefig(DocumentAddress+'/High_%d_%.3f_%.3f'%(newtime,posx,posy)+'.svg',bbox_inches='tight',format='svg',dpi=600)
                    plt.close('all')
                    plt.figure(figsize=(5,5))
                    sub_image(111,image_low,num_grid) 
                    plt.savefig(DocumentAddress+'/Low_%d_%.3f_%.3f'%(newtime,posx,posy)+'.svg',bbox_inches='tight',format='svg',dpi=600)
                    plt.close('all')
                    plt.figure(figsize=(5,5))
                    sub_image(111,image_n,num_grid) 
                    plt.savefig(DocumentAddress+'/N_%d_%.3f_%.3f'%(newtime,posx,posy)+'.svg',bbox_inches='tight',format='svg',dpi=600)
                    plt.close('all')
                    plt.figure(figsize=(5,5))
                    sub_image(111,image_n>image_n_std,num_grid) 
                    plt.savefig(DocumentAddress+'/Nos_%d_%.3f_%.3f'%(newtime,posx,posy)+'.svg',bbox_inches='tight',format='svg',dpi=600)
                    plt.close('all')
                    plt.figure(figsize=(5,5))
                    sub_image(111,image_cluster,num_grid) 
                    plt.savefig(DocumentAddress+'/Object_%d_%.3f_%.3f'%(newtime,posx,posy)+'.svg',bbox_inches='tight',format='svg',dpi=600)
                    plt.close('all')
                    plt.figure(figsize=(5,5))
                    sub_image(111,image_gain,num_grid) 
                    plt.savefig(DocumentAddress+'/Gain_%d_%.3f_%.3f'%(newtime,posx,posy)+'.svg',bbox_inches='tight',format='svg',dpi=600)
                    plt.close('all')
                    plt.figure(figsize=(5,5))
                    sub_image(111,globalmap,num_out_x) 
                    plt.savefig(DocumentAddress+'/Global_%d_%.3f_%.3f'%(newtime,posx,posy)+'.svg',bbox_inches='tight',format='svg',dpi=600)
                    plt.close('all')
                    globalmap_level[np.where(globalmap>=minlevel_map)] = 1
                    plt.figure(figsize=(5,5))
                    sub_image(111,globalmap_level,num_out_x) 
                    plt.savefig(DocumentAddress+'/GlobalLevel_%d_%.3f_%.3f'%(newtime,posx,posy)+'.svg',bbox_inches='tight',format='svg',dpi=600)
                    plt.close('all')
                    #plt.savefig(DocumentAddress+'/Map&Cluster_'+time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime())+'.png')
    except (KeyboardInterrupt,Exception) as e:
        dev.close()
        image_csvfile.close()
        obj_csvfile.close()
        raise
    else:
        dev.close()
        image_csvfile.close()
        obj_csvfile.close()
    finally:
        pass