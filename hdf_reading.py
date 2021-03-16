from __future__ import division
import math
from pyhdf import HDF
from pyhdf.SD import *
import numpy as np
import pandas as pd
import re
import scipy as sp
import scipy.signal as sig
from pyhdf.VS import *
import cv2
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import mean_proccess

LZU_LatLon = [35.946, 104.137]


def Decode_UTC(utc):
    yymmdd = utc // 1
    time = utc % 1
    time = time * 86400
    hh = int(time // 3600)
    mm = int(time % 3600 // 60 // 1)
    ss = int(time % 3600 % 60 // 1)
    hhmmss = str(hh[0])[:2] + ':' + str(mm[0])[:2] + ':' + str(ss[0])[:2]
    return yymmdd, hhmmss


def LonLat_Distance(lonlat1, lonlat2):
    r_earth = 6378.2064
    d_lonlat = math.acos((math.sin(lonlat1[0] * math.pi / 180) * math.sin(lonlat2[0] * math.pi / 180)) +
                         (math.cos(lonlat1[0] * math.pi / 180) * math.cos(lonlat2[0] * math.pi / 180) *
                          math.cos(lonlat1[1] * math.pi / 180 - lonlat2[1] * math.pi / 180))) * r_earth
    return d_lonlat


def Data_dic_select(dic, _min, _max):
    l_Rd_dic = {}
    for key in dic:
        l_Rd_dic[key] = dic[key].loc[:, dic[key].columns < _max]
        l_Rd_dic[key] = l_Rd_dic[key].loc[:, l_Rd_dic[key].columns > _min]
    return l_Rd_dic


def L2_VFM_Reading(fpath):
    sd_obj = SD(fpath, SDC.READ)
    Vt_obj = HDF.HDF(fpath).vstart()
    m_data = Vt_obj.attach('metadata').read()[0]
    Height = np.array(m_data[-1])  # 583高度对应实际海拔
    Lats = sd_obj.select('Latitude').get()
    Lons = sd_obj.select('Longitude').get()
    L_route = np.concatenate([Lats.T, Lons.T]).T
    target_rows = []

    for location in L_route:
        distance = LonLat_Distance(location, LZU_LatLon)
        if distance < 50:
            target_rows.append(True)
        else:
            target_rows.append(False)

    VFM_basic = np.array(sd_obj.select('Feature_Classification_Flags').get())
    VFM_basic = VFM_basic % 8
    VFM_1 = np.reshape(VFM_basic[:, 0:165], (VFM_basic.shape[0] * 3, 55))
    VFM_1 = np.repeat(VFM_1, 5, axis=0)
    VFM_2 = np.reshape(VFM_basic[:, 165:1165], (VFM_basic.shape[0] * 5, 200))
    VFM_2 = np.repeat(VFM_2, 3, axis=0)
    VFM_3 = np.reshape(VFM_basic[:, 1165:5515], (VFM_basic.shape[0] * 15, 290))
    VFM = np.concatenate((VFM_1, VFM_2, VFM_3), axis=1)
    target_rows_VFM = np.repeat(target_rows, 15)
    Rd_dic = {}
    Rd_dic['VFM'] = VFM
    Rd_dic_meta = {
        'route': L_route,
        'Lats': Lats,
        'target rows': target_rows,
        'Height': Height,
        'target rows VFM': target_rows_VFM,
    }
    sd_obj.end()
    HDF.HDF(fpath).close()
    return Rd_dic, Rd_dic_meta


# 获取文件内数据字典
def L1_Reading(fpath):
    sd_obj = SD(fpath, SDC.READ)
    Vt_obj = HDF.HDF(fpath).vstart()
    m_data = Vt_obj.attach('metadata').read()[0]
    Height = np.array(m_data[-2])  # 583高度对应实际海拔
    Lats = sd_obj.select('Latitude').get()
    Lons = sd_obj.select('Longitude').get()
    L_route = np.concatenate([Lats.T, Lons.T]).T
    del Lons
    surface = sd_obj.select('Surface_Elevation').get()
    target_rows = []
    distance_list = []
    min_distance = 9999999
    for location in L_route:
        distance = LonLat_Distance(location, LZU_LatLon)
        if distance < min_distance:
            min_distance = distance
        if distance < 50:
            target_rows.append(True)
        else:
            target_rows.append(False)
        distance_list.append(distance)

    kernel = []
    aPer532 = np.array(sd_obj.select('Perpendicular_Attenuated_Backscatter_532').get())
    Per532 = cv2.blur(aPer532, (1, 15))
    Per532[Per532 < 0] = 0
    aTol532 = np.array(sd_obj.select('Total_Attenuated_Backscatter_532').get())
    Tol532 = cv2.blur(aTol532, (1, 15))
    Tol532[Tol532 < 0] = 0
    aPar532 = Tol532 - Per532
    Par532 = cv2.blur(aPar532, (1, 15))
    # proccess Dep data
    Dep532 = np.true_divide(Per532, Par532)
    Dep532[Par532 <= 0.0003] = np.nan
    Dep532[Par532 <= 0.0000] = 0
    Dep532[Dep532 > 1] = np.nan
    Data_dic = {}
    Data_dic['Tol532'] = Tol532
    # Data_dic['Per532'] = Per532
    # Data_dic['Par532'] = Par532
    Data_dic['Dep532'] = Dep532
    Data_meta = {
        'route': L_route,
        'surface': surface,
        'Lats': Lats,
        'target rows': target_rows,
        'Height': Height,
        'distance': distance_list,
        'min distance': min_distance
    }
    # for key, value in Rd_dic.items():
    # value.columns = Height.values[0]
    sd_obj.end()
    HDF.HDF(fpath).close()
    return Data_dic, Data_meta


def L1_VFM_proccess(f_path, vfm_path):
    print('hi')
    L1_dic, L1_meta = L1_Reading(f_path)
    VFM_dic, VFM_meta = L2_VFM_Reading(vfm_path)
    L1_frame_dic = {}
    clear_L1_Data = {}
    target_VFM = {}
    target_L1 = {}
    target_route = VFM_meta['route'][VFM_meta['target rows']]
    if target_route[0][0] < target_route[-1][0]:
        loc_range = [target_route[0][0], target_route[-1][0]]
    else:
        loc_range = [target_route[-1][0], target_route[0][0]]
    del target_route
    ttt = (loc_range[0] <= L1_meta['Lats']) & (loc_range[1] >= L1_meta['Lats'])
    fff = ttt.copy()
    for i in range(len(ttt)):
        if ttt[i][0]:
            fff[i + 14][0] = True
    for key in L1_dic:
        target_L1[key] = L1_dic[key][fff.T[0]]
    for key in VFM_dic:
        target_VFM[key] = VFM_dic[key][VFM_meta['target rows VFM']]

    target_route = np.array(L1_meta['route'])[fff.T[0]]
    target_distance = np.array(L1_meta['distance'])[fff.T[0]]
    target_surface = np.array(L1_meta['surface'])[fff.T[0]]
    min_point = np.where(target_distance == L1_meta['min distance'])[0][0]
    cloud_status = []
    target_route_str = []
    for i in range(target_route.shape[0]):
        target_route_str.append(str(format(target_route[i][0], '.2f')) + '\n' + str(format(target_route[i][1], '.2f')))

    for j in target_VFM['VFM']:
        if 2 in target_VFM['VFM'][j]:
            cloud_status.append(False)
        else:
            cloud_status.append(True)

    for keys in target_L1:
        L1_frame_dic[keys] = pd.DataFrame(target_L1[keys], columns=L1_meta['Height'])
        clear_L1_Data[keys] = pd.DataFrame(target_L1[keys][cloud_status], columns=L1_meta['Height'])

    '''    VFM_frame = pd.DataFrame(target_VFM['VFM'],
                             columns=L1_meta['Height'],
                             index=target_distance)'''
    Avg_Rd = {}
    for keys in L1_frame_dic:
        Avg_Rd[keys] = np.nanmean(L1_frame_dic[keys].values, axis=0)
        Avg_Rd[keys] = mean_proccess.mean5_3(Avg_Rd[keys], 5)

    clr_Avg_Rd = {}
    for keys in clear_L1_Data:
        clr_Avg_Rd[keys] = np.nanmean(clear_L1_Data[keys].values, axis=0)
        clr_Avg_Rd[keys] = mean_proccess.mean5_3(clr_Avg_Rd[keys], 5)

    return Avg_Rd['Dep532'], clr_Avg_Rd['Dep532'], L1_meta['Height'], L1_meta['min distance'], \
           target_L1['Dep532'], target_L1['Tol532'], target_route_str, target_surface, min_point