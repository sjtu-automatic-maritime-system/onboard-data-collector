#!/usr/bin/python
# -*- coding: utf-8 -*-

#Output:'globalmap','objects','globalmap_size','objnum';Port number:55012
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import csv
from math import*
from msgdev import MsgDevice,PeriodTimer

#Period settings
t_refresh = 1 #period to refresh

#Chart settings
x_chart = np.array([-155,45]) #x range in chart
y_chart = np.array([-100,100]) #x range in chart
res_chart  = 1 #grid size or resolution in chart
chart = np.zeros((ceil((x_chart[1]-x_chart[0])/res_chart),ceil((y_chart[1]-y_chart[0])/res_chart)))
chart[149:200] = 1 #initialize chart
chart[:,0:25] = 1
chart[:,175:200] = 1

#Global map settings
x_global = np.array([-155,45]) #x range in global map
y_global = np.array([-100,100]) #y range in global map
res_global  = 1 #grid size or resolution in global map

#Image Settings
image_size = 50 #side length of a square image
grid_size = 0.2 #grid size in image
vrange = 5 #vertical value range from -vrange to vrange

#Cluster Settings
minarea_cluster = 3
neighborsize = 2
mindensity_normal = 3
mindensity_fast = 6

#Object Settings
maxlen_obj = 4
maxlow_obj = 1.5

#Cluster to object settings
c2o_hlimit = 0.5 #maximum difference in h
c2o_alimit = 5 #maximum difference in area
c2o_rlimit  = 0.2 #maximum difference in B/L
c2o_vlimit = 3 ##maximum velocity

#Object to map settings
o2m_dlimit = 1 #maximum moving distance of a static obstacle

#Confidence level settings
maxlevel_obj = 24
minlevelgood_obj = 4
maxlevel_map = 20
minlevel_map = 10 #minimum confidence level of static obstacles in global map

#Kalman Filter of vx,vy settings
kf_p0 = 1
kf_q = 0.25
kf_r = 1

# Parameter Calculation
num_grid = ceil(image_size/grid_size)
num_out_x = ceil((x_global[1]-x_global[0])/res_global)
num_out_y = ceil((y_global[1]-y_global[0])/res_global)
num_trans = ceil(res_global/grid_size)
num_chartrans = ceil(res_chart/grid_size)
vlp_frequency = 10
res_vlp = vlp_frequency*2/100/57.3

#Standard number image settings
image_n_std = np.zeros((num_grid,num_grid))
for p in range(num_grid):
    for q in range(num_grid):
        if p!=(num_grid/2) or q!=(num_grid/2):
            dis_pq = grid_size*sqrt((p-num_grid/2)**2+(q-num_grid/2)**2)
            n_std_pq = ceil(grid_size/(dis_pq*res_vlp))
            image_n_std[p,q] = n_std_pq*0.2


def devinitial(dev):
    dev.pub_bind('tcp://0.0.0.0:55012')
    dev.sub_connect('tcp://127.0.0.1:55011')
    dev.sub_add_url('image.image_high',[0,]*(num_grid*num_grid))
    dev.sub_add_url('image.image_low',[0,]*(num_grid*num_grid))
    dev.sub_add_url('image.image_n',[0,]*(num_grid*num_grid))
    dev.sub_add_url('image.runtime')
    dev.sub_add_url('image.posx')
    dev.sub_add_url('image.posy')

def pca(coor):
    data_adj = coor - np.mean(coor,axis=0)
    cov = np.cov(data_adj, rowvar=False)
    eigVals,eigVects = np.linalg.eig(np.mat(cov))
    eigValInd = np.argsort(eigVals)
    redEigVects = eigVects[:,eigValInd]
    mainvec0 = redEigVects[0,1]
    mainvec1 = redEigVects[1,1]
    if mainvec0 == 0:
        heading = pi/2
    else:
        heading  = atan(mainvec1/mainvec0)
    return data_adj.dot(redEigVects),heading

def kf(vx_old,vy_old,vx_sensor,vy_sensor,var_old):
    k = (var_old+kf_q)/(var_old+kf_q+kf_r)
    vx = (1-k)*vx_old + k*vx_sensor
    vy = (1-k)*vy_old + k*vy_sensor
    var = (1-k)*(var_old+kf_q)
    return vx,vy,var

def sub_image(subnum,data,title,picsize,cmaptype='viridis'):
    plt.subplot(subnum)
    img  = data*1
    for i in range(picsize):
        img[:,i] = img[:,i][::-1]
    imgplot = plt.imshow(img,cmap=cmaptype)
    plt.xlabel('y')
    plt.ylabel('x')
    plt.colorbar()
    plt.title(title)


class Map:
    def __init__(self):
        self.globalchart = np.zeros((ceil((x_chart[1]-x_chart[0])/grid_size),ceil((y_chart[1]-y_chart[0])/grid_size))) #initialize global chart
        for i in range(ceil((x_chart[1]-x_chart[0])/grid_size)):
            for j in range(ceil((y_chart[1]-y_chart[0])/grid_size)):
                self.globalchart[i,j] = chart[i//num_chartrans,j//num_chartrans]
        self.globalmap = self.globalchart*minlevel_map #initialize global static obstacle map
        self.globalmap_out = np.zeros((num_out_x,num_out_y)) #initialize output global static obstacle map
        self.objects = np.zeros((0,17)) #[area,x_min,x_max,y_min,y_max,l,b,heading,h,vx,vy,var_v,dis_x,dis_y,confidence level,cluster_num,object label]
        self.localmap  = np.zeros((num_grid,num_grid))
        self.runtime = 0
        self.num_pub = 0
        self.objectlabel = 0
        print('Global Map Created.')

    def get_localmap(self,posx,posy):
        self.localmap = self.globalmap[floor((posx-x_global[0])/grid_size-num_grid/2):floor((posx-x_global[0])/grid_size+num_grid/2),floor((posy-y_global[0])/grid_size-num_grid/2):floor((posy-y_global[0])/grid_size+num_grid/2)]
        self.localchart = self.globalchart[floor((posx-x_global[0])/grid_size-num_grid/2):floor((posx-x_global[0])/grid_size+num_grid/2),floor((posy-y_global[0])/grid_size-num_grid/2):floor((posy-y_global[0])/grid_size+num_grid/2)]

    def confirm_localmap(self): #if the image has obstacles where local map has obstacles, confidence level will increase, otherwise it will decrease
        for p in range(num_grid):
            for q in range(num_grid):
                if self.localmap[p,q] > 0 and self.image_binary[p,q] == 0:
                    self.localmap[p,q] -= 1

    def find_cluster(self):
        #if obstacles have good confidence level in local map or exist in chart, they will be marked with a very large cluster number 10000
        fast_cluster = np.zeros((num_grid,num_grid))
        fast_cluster[np.where(self.localmap>=minlevel_map)] = 10000
        fast_cluster[np.where(self.localchart==1)] = 10000
        #DBSCAN density set to mindensity_fast,which is larger than mindensity_normal, to find cluster 10000
        fast_ctest = (fast_cluster>0)*1
        fast_cstate = fast_ctest*1
        self.num_cluster = 0
        for p in range(num_grid):
            for q in range(num_grid):
                if self.image_binary[p,q] == 1 and (not(fast_ctest[p,q]==1 and fast_cstate[p,q]==0)):
                    neighbor_xmin = max(0,p-neighborsize)
                    neighbor_xmax = min(num_grid,p+neighborsize+1)
                    neighbor_ymin = max(0,q-neighborsize)
                    neighbor_ymax = min(num_grid,q+neighborsize+1)
                    neighbor_binary = self.image_binary[neighbor_xmin:neighbor_xmax,neighbor_ymin:neighbor_ymax]
                    neighbor_num = neighbor_binary.sum()
                    fast_ctest[p,q] = 1
                    if neighbor_num >= mindensity_fast:
                        fast_cstate[p,q] = 1
                        max_clusternum = (fast_cluster[neighbor_xmin:neighbor_xmax,neighbor_ymin:neighbor_ymax]*fast_cstate[neighbor_xmin:neighbor_xmax,neighbor_ymin:neighbor_ymax]).max() #max cluster number of fast core points in neighbor
                        #no fast core in  belong to existing cluster
                        if max_clusternum == 0 :
                            self.num_cluster += 1
                            fast_cluster[p,q] = self.num_cluster
                            for m in range(neighbor_xmin,neighbor_xmax):
                                for n in range(neighbor_ymin,neighbor_ymax):
                                    if self.image_binary[m,n]==1:
                                        fast_cluster[m,n] = self.num_cluster
                                        if fast_ctest[m,n]==0:
                                            mn_neighbor_xmin = max(0,m-neighborsize)
                                            mn_neighbor_xmax = min(num_grid,m+neighborsize+1)
                                            mn_neighbor_ymin = max(0,n-neighborsize)
                                            mn_neighbor_ymax = min(num_grid,n+neighborsize+1)
                                            mn_neighbor_binary = self.image_binary[mn_neighbor_xmin:mn_neighbor_xmax,mn_neighbor_ymin:mn_neighbor_ymax]
                                            mn_neighbor_num = mn_neighbor_binary.sum()
                                            fast_ctest[m,n] = 1
                                            if mn_neighbor_num >= mindensity_fast:
                                                fast_cstate[m,n] = 1
                        else:
                            fast_cluster[p,q] = max_clusternum
                            for m in range(neighbor_xmin,neighbor_xmax):
                                for n in range(neighbor_ymin,neighbor_ymax):
                                    if self.image_binary[m,n]==1:
                                        if fast_cluster[m,n] == 0:
                                            fast_cluster[m,n] = max_clusternum
                                            if fast_ctest[m,n]==0:
                                                mn_neighbor_xmin = max(0,m-neighborsize)
                                                mn_neighbor_xmax = min(num_grid,m+neighborsize+1)
                                                mn_neighbor_ymin = max(0,n-neighborsize)
                                                mn_neighbor_ymax = min(num_grid,n+neighborsize+1)
                                                mn_neighbor_binary = self.image_binary[mn_neighbor_xmin:mn_neighbor_xmax,mn_neighbor_ymin:mn_neighbor_ymax]
                                                mn_neighbor_num = mn_neighbor_binary.sum()
                                                fast_ctest[m,n] = 1
                                                if mn_neighbor_num >= mindensity_fast:
                                                    fast_cstate[m,n] = 1
                                        elif fast_cluster[m,n]!=max_clusternum:
                                            if fast_cstate[m,n] == 1:#only core points can combine their cluster with others
                                                fast_cluster[np.where(fast_cluster==fast_cluster[m,n])] = max_clusternum
        fast_cluster[np.where(fast_cluster!=10000)] = 0
        #find normal cluster without disturbance of cluster 10000
        self.image_binary[np.where(fast_cluster==10000)] = 0
        #DBSCAN density set to mindensity_normal to find normal cluster
        normal_cluster = np.zeros((num_grid,num_grid))
        normal_ctest = np.zeros((num_grid,num_grid))
        normal_cstate = np.zeros((num_grid,num_grid))
        self.num_cluster = 0
        for p in range(num_grid):
            for q in range(num_grid):
                if self.image_binary[p,q] == 1 and (not(normal_ctest[p,q]==1 and normal_cstate[p,q]==0)):
                    neighbor_xmin = max(0,p-neighborsize)
                    neighbor_xmax = min(num_grid,p+neighborsize+1)
                    neighbor_ymin = max(0,q-neighborsize)
                    neighbor_ymax = min(num_grid,q+neighborsize+1)
                    neighbor_binary = self.image_binary[neighbor_xmin:neighbor_xmax,neighbor_ymin:neighbor_ymax]
                    neighbor_num = neighbor_binary.sum()
                    normal_ctest[p,q] = 1
                    if neighbor_num >= mindensity_normal:
                        normal_cstate[p,q] = 1
                        max_clusternum = (normal_cluster[neighbor_xmin:neighbor_xmax,neighbor_ymin:neighbor_ymax]*normal_cstate[neighbor_xmin:neighbor_xmax,neighbor_ymin:neighbor_ymax]).max() #max cluster number in neighbor
                        #no neighbor belong to existing cluster
                        if max_clusternum == 0 :
                            self.num_cluster += 1
                            normal_cluster[p,q] = self.num_cluster
                            for m in range(neighbor_xmin,neighbor_xmax):
                                for n in range(neighbor_ymin,neighbor_ymax):
                                    if self.image_binary[m,n]==1:
                                        normal_cluster[m,n] = self.num_cluster
                                        if normal_ctest[m,n]==0:
                                            mn_neighbor_xmin = max(0,m-neighborsize)
                                            mn_neighbor_xmax = min(num_grid,m+neighborsize+1)
                                            mn_neighbor_ymin = max(0,n-neighborsize)
                                            mn_neighbor_ymax = min(num_grid,n+neighborsize+1)
                                            mn_neighbor_binary = self.image_binary[mn_neighbor_xmin:mn_neighbor_xmax,mn_neighbor_ymin:mn_neighbor_ymax]
                                            mn_neighbor_num = mn_neighbor_binary.sum()
                                            normal_ctest[m,n] = 1
                                            if mn_neighbor_num >= mindensity_normal:
                                                normal_cstate[m,n] = 1
                        else:
                            normal_cluster[p,q] = max_clusternum
                            for m in range(neighbor_xmin,neighbor_xmax):
                                for n in range(neighbor_ymin,neighbor_ymax):
                                    if self.image_binary[m,n]==1:
                                        if normal_cluster[m,n] == 0:
                                            normal_cluster[m,n] = max_clusternum
                                            if normal_ctest[m,n]==0:
                                                mn_neighbor_xmin = max(0,m-neighborsize)
                                                mn_neighbor_xmax = min(num_grid,m+neighborsize+1)
                                                mn_neighbor_ymin = max(0,n-neighborsize)
                                                mn_neighbor_ymax = min(num_grid,n+neighborsize+1)
                                                mn_neighbor_binary = self.image_binary[mn_neighbor_xmin:mn_neighbor_xmax,mn_neighbor_ymin:mn_neighbor_ymax]
                                                mn_neighbor_num = mn_neighbor_binary.sum()
                                                normal_ctest[m,n] = 1
                                                if mn_neighbor_num >= mindensity_normal:
                                                    normal_cstate[m,n] = 1
                                        elif normal_cluster[m,n]!=max_clusternum: #only core points can combine their cluster with others
                                            if normal_cstate[m,n] == 1:
                                                normal_cluster[np.where(normal_cluster==normal_cluster[m,n])] = max_clusternum
        self.image_cluster = fast_cluster + normal_cluster
        #remove noise
        self.image_binary[np.where(self.image_cluster==0)] = 0

    def analyze_cluster(self,posx,posy):
        self.property_cluster = np.zeros((0,10))
        for k in range(1,self.num_cluster+1):
            location = np.where(self.image_cluster==k)
            area_cluster = len(location[0])
            if area_cluster<minarea_cluster: #some clusters were absorbed by others, some are too small to be an object
                self.image_cluster[location] = 0
                self.num_cluster -=1
                self.image_binary[location] = 1
            else: #calculate basic properties of each cluster
                h_cluster = self.image_high[location].max()
                low_cluster  = self.image_low[location].min()
                coor_cluster  = np.column_stack((location[0],location[1]))
                pca_cluster,heading_cluster  = pca(coor_cluster)
                l_cluster  = (pca_cluster[:,1].max()-pca_cluster[:,1].min())*grid_size
                b_cluster  = (pca_cluster[:,0].max()-pca_cluster[:,0].min())*grid_size
                if l_cluster > maxlen_obj:
                    self.image_cluster[location] = 0
                    self.num_cluster -=1
                    self.image_binary[location] = ceil(minlevel_map/4)
                elif low_cluster > maxlow_obj:
                    self.image_cluster[location] = 0
                    self.num_cluster -=1
                    self.image_binary[location] = 0
                else:
                    self.property_cluster = np.row_stack((self.property_cluster,np.array([area_cluster,(location[0].min()-num_grid/2)*grid_size+posx,(location[0].max()-num_grid/2)*grid_size+posx,(location[1].min()-num_grid/2)*grid_size+posy,(location[1].max()-num_grid/2)*grid_size+posy,l_cluster,b_cluster,heading_cluster,h_cluster,k])))

    def c2o(self): #match clusters and objects
        len_obj = len(self.objects)
        match_c = np.zeros(self.num_cluster)
        for q in range(len_obj):
            q_objects = self.objects[q]
            q_find = 12
            for p in range(self.num_cluster):
                if match_c[p] == 0:
                    p_cluster = self.property_cluster[p]
                    p2q_x = (p_cluster[1]+p_cluster[2]-q_objects[1]-q_objects[2])/2
                    p2q_y = (p_cluster[3]+p_cluster[4]-q_objects[3]-q_objects[4])/2
                    p2q_x_error = p2q_x - q_objects[9]*self.interval
                    p2q_y_error = p2q_y - q_objects[10]*self.interval
                    p2q_v = sqrt(p2q_x_error**2+p2q_y_error**2)
                    p2q_a = abs(p_cluster[0]-q_objects[0])
                    p2q_h = abs(p_cluster[8]-q_objects[8])
                    p2q_r = abs(p_cluster[6]/p_cluster[5]-q_objects[6]/q_objects[5])
                    p2q_judge = p2q_a/c2o_alimit + p2q_r/c2o_rlimit + p2q_h/c2o_hlimit + 2*p2q_v/c2o_vlimit
                    if p2q_judge<q_find:
                        q_find = p2q_judge
                        q_find_pos = p
                        q_find_p2q_x=p2q_x
                        q_find_p2q_y=p2q_y
            if q_find == 12:
                self.objects[q,14] = self.objects[q,14]-1
                self.objects[q,1] = self.objects[q,1] + self.objects[q,9]*self.interval
                self.objects[q,2] = self.objects[q,2] + self.objects[q,9]*self.interval
                self.objects[q,12] = self.objects[q,12] + self.objects[q,9]*self.interval
                self.objects[q,3] = self.objects[q,3] + self.objects[q,10]*self.interval
                self.objects[q,4] = self.objects[q,4] + self.objects[q,10]*self.interval
                self.objects[q,13] = self.objects[q,13] + self.objects[q,10]*self.interval
                self.objects[q,11] = self.objects[q,11] + kf_q
            else:
                match_c[q_find_pos] = 1
                p_cluster = self.property_cluster[q_find_pos]
                self.objects[q,0:9] = p_cluster[0:9]
                self.objects[q,9],self.objects[q,10],self.objects[q,11] = kf(self.objects[q,9],self.objects[q,10],q_find_p2q_x/self.interval,q_find_p2q_y/self.interval,self.objects[q,11]) #use kf to filter vx and vy
                self.objects[q,12] = self.objects[q,12]+q_find_p2q_x
                self.objects[q,13] = self.objects[q,13]+q_find_p2q_y
                self.objects[q,14] = self.objects[q,14]+2
                self.objects[q,15] = p_cluster[9]
        for p in range(self.num_cluster):
            if match_c[p] == 0:
                new_obj = np.zeros(17)
                new_obj[0:9] = self.property_cluster[p,0:9]
                new_obj[9] = 0
                new_obj[10] = 0
                new_obj[11] = kf_p0
                new_obj[12] = 0
                new_obj[13] = 0
                new_obj[14] = 2
                new_obj[15] = self.property_cluster[p,9]
                self.objectlabel += 1
                new_obj[16] = self.objectlabel
                self.objects = np.row_stack((self.objects,new_obj))
        self.objects[:,14] = self.objects[:,14].clip(0,maxlevel_obj)
        self.objects = np.delete(self.objects,np.where(self.objects[:,14]==0),axis=0)

    def o2m(self): #judge if any object can be regarded as static obstacle
        self.image_binary[np.where(self.image_cluster!=0)] = 0 #objects will not increase confidence level in global map
        self.image_binary[np.where(self.image_cluster==10000)] = minlevel_map/2 #obstacles with cluster number 10000 will increase confidence level very quickly in global map
        for k in range(len(self.objects)):
            if (self.objects[k,14] >= maxlevel_obj/2) and ((abs(self.objects[k,12])+abs(self.objects[k,13])) < o2m_dlimit):
                self.objects[k,14] += 100
                self.image_binary[np.where(self.image_cluster==self.objects[k,15])] = minlevel_map
        self.objects = np.delete(self.objects,np.where(self.objects[:,14]>maxlevel_obj),axis=0)
        self.image_cluster[np.where(self.image_cluster==10000)] = 0
        self.objects = self.objects[np.argsort(-self.objects[:,14])]

    def refresh_localmap(self):
        self.localmap = self.image_binary + self.localmap
        self.localmap = self.localmap.clip(0,maxlevel_map)

    def refresh_globalmap(self,posx,posy):
        self.globalmap[floor((posx-x_global[0])/grid_size-num_grid/2):floor((posx-x_global[0])/grid_size+num_grid/2),floor((posy-y_global[0])/grid_size-num_grid/2):floor((posy-y_global[0])/grid_size+num_grid/2)] = self.localmap
        for i in range(num_out_x):
            for j in range(num_out_y):
                self.globalmap_out[i,j] = self.globalmap[i*num_trans:(i+1)*num_trans,j*num_trans:(j+1)*num_trans].max()

    def update(self,posx,posy,image_high,image_low,image_n,runtime,dev):
        newtime = runtime
        self.interval = newtime - self.runtime
        print('interval:' , self.interval)
        if self.interval != 0:
            self.runtime = newtime
            self.image_high = image_high
            self.image_low = image_low
            self.image_n = image_n
            self.image_binary = (image_n >= image_n_std)*1
            self.get_localmap(posx,posy)
            self.confirm_localmap()
            self.find_cluster()
            self.analyze_cluster(posx,posy)
            self.c2o()
            self.o2m()
            self.refresh_localmap()
            self.refresh_globalmap(posx,posy)
            self.publish(dev,posx,posy)

    def publish(self,dev,posx,posy):
        #publish position information
        inf = [posx,posy,self.runtime]
        print('Runtime Map:',self.runtime)
        dev.pub_set('map.pos&time',inf)

        #publish image information
        dev.pub_set('map.image_high',self.image_high.flatten().tolist())
        dev.pub_set('map.image_low',self.image_low.flatten().tolist())
        dev.pub_set('map.image_n',self.image_n.flatten().tolist())
        dev.pub_set('map.image_cluster',self.image_cluster.flatten().tolist())
        dev.pub_set('map.image_gain',self.image_binary.flatten().tolist())

        #publish globalmap information
        dev.pub_set('map.globalmap',self.globalmap_out.flatten().tolist())
        globalmap_size = [num_out_x,x_global[0],x_global[1],num_out_y,y_global[0],y_global[1]]
        dev.pub_set('map.globalmap_size',globalmap_size)

        #publish object information
        objnum = len(self.objects)
        dev.pub_set1('map.objnum',objnum)
        objects_out = np.zeros((100,17))
        objects_out[0:objnum,:] = self.objects
        dev.pub_set('map.objects',objects_out.flatten().tolist())

        #publish time information
        self.num_pub += 1
        dev.pub_set1('map.timestamp',self.num_pub)
        #print('Global Map Range:',self.globalmap_out.min(),'-',self.globalmap_out.max(),'\nObject Number:',objnum)

if __name__ == "__main__":
    try:
        dev  = MsgDevice()
        dev.open()
        devinitial(dev)
        mymap = Map()
        t = PeriodTimer(t_refresh)
        t.start()
        while True:
            with t:
                image_high = np.fromiter(dev.sub_get('image.image_high'),'d').reshape(num_grid,num_grid)
                image_low = np.fromiter(dev.sub_get('image.image_low'),'d').reshape(num_grid,num_grid)
                image_n = np.fromiter(dev.sub_get('image.image_n'),'d').reshape(num_grid,num_grid)
                posx = dev.sub_get1('image.posx')
                posy = dev.sub_get1('image.posy')
                runtime = dev.sub_get1('image.runtime')
                mymap.update(posx,posy,image_high,image_low,image_n,runtime,dev)
                #print('No.%d'%mymap.num_pub)

    except (KeyboardInterrupt,Exception) as e:
        dev.close()
        raise
    else:
        dev.close()
    finally:
        pass


