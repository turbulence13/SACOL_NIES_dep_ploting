import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import re
import hdf_reading as hr
import mean_proccess as mp
import color_data as clrd


def allspines_set(ax, is_on=True, width=1):  # 坐标轴线格式
    if is_on:
        for spine in ax.spines:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(width)
    else:
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)


def Radar_heat(data_dic, time_area=None, height_area=None):  # 针对本次绘图设计绘图函数，应当针对性进行改变
    x_minorlocator = AutoMinorLocator(n=3)
    y_ticks = np.linspace(0, 1677, 6)
    y_label = ('0.0', '2.0', '4.0', '6.0', '8.0', '10.0')
    f, ax = plt.subplots(nrows=len(data_dic), sharex=True, figsize=(6, 6))
    i = 0
    ax_dic = {}
    for key in data_dic:
        ax_dic[key] = ax[i]
        i = i + 1
    sns.heatmap(data_dic['It532'], vmax=40.0, vmin=0.0, cmap=clrd.custom,
                ax=ax_dic['It532'], yticklabels=400, xticklabels=18)
    ax_dic['It532'].invert_yaxis()
    ax_dic['It532'].set_xticks(np.linspace(0, 1440, 8))
    sns.heatmap(data_dic['Dp532'], vmax=0.5, vmin=0.0, cmap=clrd.custom,
                ax=ax_dic['Dp532'], yticklabels=400, xticklabels=18)
    ax_dic['Dp532'].invert_yaxis()
    ax_dic['Dp532'].set_xlabel('Time')
    if (time_area is not None) & (height_area is not None):
        height_c = height_area.copy()
        height_c[0] = height_c[0] * 166.6666
        height_c[1] = height_c[1] * 166.6666
        for keys in ax_dic:  # 坐标轴刻度格式
            ax_dic[keys].vlines(time_area, ymin=height_c[0], ymax=height_c[1], colors='black',
                                linestyles='dashed')
            ax_dic[keys].hlines(height_c, xmin=time_area[0], xmax=time_area[1], colors='black',
                                linestyles='dashed')
    for keys in ax_dic:  # 坐标轴刻度格式
        ax_dic[keys].set_yticks(y_ticks)
        ax_dic[keys].set_yticklabels(y_label, rotation=0)
        ax_dic[keys].minorticks_on()
        ax_dic[keys].xaxis.set_minor_locator(x_minorlocator)
        allspines_set(ax_dic[keys], width=1)  # 坐标轴框线


def dep_by_height(data, meantime=1, top=10.0, bottum=0.0):
    data_a = data.copy()
    # data_a[np.isnan(data_a)] = 0
    # data_a[np.isinf(data_a)] = 0
    data_b = np.nanmean(data_a, axis=1)
    data_b[data_b < 0] = 0
    data_b = mp.mean_simple(data_b, meantime)
    data_c = pd.DataFrame(data=data_b, index=data.index)
    avg_data = np.nanmean(data_c.loc[(data_c.index <= top) & (data_c.index >= bottum)].values)
    return data_c, avg_data


def plot_by_height(series, top=10.0, bottum=0.0, horizontal=None):
    if horizontal is None:
        horizontal = [0, 0.1]
    plt.figure(figsize=(3, 4.5))
    plt.axis([horizontal[0], horizontal[1], top, bottum])
    plt.plot(series.values, series.index, color='blue', linewidth=1.0)
    # fig.xticks(np.linspace(0, 1440, 8))


def date_files_reading(date, path):
    files = ('SACOL_NIESLIDAR_' + date + '_Int532_Dep532_Int1064.dat')
    os.chdir(path)
    f_data = pd.read_table(files, sep='\s+', index_col='Height(km)', na_values=['NaN'], skiprows=3)
    data = {
        'It532': f_data.iloc[0:3000][:],
        'Dp532': f_data.iloc[3000:6000][:],
    }
    return data


def date_L1_reading(date, path, path_vfm):
    os.chdir(path)
    f_list = os.listdir(path)
    t_date = date[0:4] + '-' + date[4:6] + '-' + date[6:8]
    for files in f_list:
        fname = re.match('^CAL_LID_L1-Standard-V4-10\.' + t_date + '.*\.hdf$', files)
        if fname is not None:
            vfm_files = path_vfm + 'CAL_LID_L2_VFM-Standard-V4-20' + files[25:]

            if hr.L1_VFM_proccess(files, vfm_files) is not None:
                Dp_height, Dp_height_clear, Data_height, min_distance, \
                Dep532_array, Tol532_array, target_route, target_surface, min_point = hr.L1_VFM_proccess(files, vfm_files)
                # Data_mean = np.nanmean(Data_dic['Dep532'], axis=0)
                target_surface = target_surface
                height = Data_height - 1.961
                # height_vfm = Data_height[(Data_height >= -0.5) & (Data_height <= 30.1)]
                Dp_Data_height = pd.DataFrame(Dp_height, index=height)
                Dep532_frame = pd.DataFrame(Dep532_array, columns=Data_height, index=target_route)
                print(min_point)
                Tol532_frame = pd.DataFrame(Tol532_array, columns=Data_height, index=target_route)
                return Dp_Data_height, min_distance, Tol532_frame, Dep532_frame, target_surface, min_point
            else:
                return None


def target_average_dp(date, path, time_area, height_area):
    # 文件读取，跳过文件说明，选取高度作为行名，便于画图
    Rddata_dic = date_files_reading(date, path)
    Rddata_dic['Dp532'].values[Rddata_dic['Dp532'].values < 0] = np.nan
    Rddata_dic['Dp532'].values[Rddata_dic['Dp532'].values > 1] = np.nan
    Dp_height, avgdata = dep_by_height(Rddata_dic['Dp532'].iloc[:, time_area[0]:time_area[1]],
                                       meantime=3, top=height_area[1], bottum=height_area[0])
    return avgdata, Dp_height


def combine_plot(Sacol_data, Dep532, VFM, Dp_height, L1_data, target_surface, min_point, min_distance,
                 time_area, height_area, calibration, horizontal):
    plt.figure(figsize=(16, 6), dpi=120)
    x_minorlocator = AutoMinorLocator(n=4)
    y1_minorlocator = AutoMinorLocator(n=3)
    y2_minorlocator = AutoMinorLocator(n=3)
    y3_minorlocator = AutoMinorLocator(n=2)
    y_ticks = np.linspace(0, 1500, 4)
    x_ticks = np.linspace(0, 144, 7)
    y2_ticks = np.linspace(0, Dep532.shape[0], 4)
    y_label = ('0.0', '3.0', '6.0', '9.0')
    x_label = ('00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00')
    height_c = height_area.copy()
    height_c[0] = height_c[0] * 166.6666
    height_c[1] = height_c[1] * 166.6666
    l_Sacol_data = {}
    for keys in Sacol_data:
        l_Sacol_data[keys] = Sacol_data[keys].loc[(Sacol_data[keys].index < 9) & (Sacol_data[keys].index > 0)]

    ax1 = plt.subplot2grid((2, 50), (0, 0), colspan=18, rowspan=1)
    sns.heatmap(l_Sacol_data['It532'], vmax=40.0, vmin=0.0, cmap=clrd.custom,
                ax=ax1, xticklabels=24)
    ax1.invert_yaxis()
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_label, rotation=0)
    ax1.set_xticks(x_ticks)
    ax1.set_ylabel('Height (AGL, km)')
    ax1.set_xticklabels(x_label, rotation=0)
    ax1.xaxis.set_minor_locator(x_minorlocator)
    ax1.yaxis.set_minor_locator(y1_minorlocator)
    allspines_set(ax1, width=1)  # 坐标轴框线
    ax1.vlines(time_area, ymin=height_c[0], ymax=height_c[1], colors='black',
               linestyles='dashed')
    ax1.hlines(height_c, xmin=time_area[0], xmax=time_area[1], colors='black',
               linestyles='dashed')
    ax1.tick_params(axis='both', labelsize=8)
    ax1.text(2, l_Sacol_data['It532'].shape[0] - 10, 'a',
             verticalalignment='top',
             horizontalalignment='left',
             fontsize=15,
             )

    ax2 = plt.subplot2grid((2, 50), (1, 0), colspan=18, rowspan=1)
    sns.heatmap(l_Sacol_data['Dp532'], vmax=0.5, vmin=0.0, cmap=clrd.custom,
                ax=ax2, xticklabels=24)
    ax2.invert_yaxis()
    ax2.set_xlabel('Time (UTC)')
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(y_label, rotation=0)
    ax2.set_xticks(x_ticks)
    ax2.set_ylabel('Height (AGL, km)')

    ax2.set_xticklabels(x_label, rotation=0)
    ax2.xaxis.set_minor_locator(x_minorlocator)
    ax2.yaxis.set_minor_locator(y1_minorlocator)
    allspines_set(ax2, width=1)  # 坐标轴框线
    ax2.vlines(time_area, ymin=height_c[0], ymax=height_c[1], colors='black',
               linestyles='dashed')
    ax2.hlines(height_c, xmin=time_area[0], xmax=time_area[1], colors='black',
               linestyles='dashed')
    ax2.tick_params(axis='both', labelsize=8)
    ax2.text(2, l_Sacol_data['Dp532'].shape[0] - 10, 'b',
             verticalalignment='top',
             horizontalalignment='left',
             fontsize=15,
             )

    ax3 = plt.subplot2grid((2, 50), (0, 39), colspan=12, rowspan=2)
    if calibration is not None:
        cal_Dp = Dp_height - calibration
    else:
        cal_Dp = Dp_height
    if horizontal is None:
        horizontal = [0, 0.1]

    ax3.axis([horizontal[0], horizontal[1], height_area[0], height_area[1]])
    ax3.plot(cal_Dp.values, cal_Dp.index, color='blue', linewidth=1.0)
    ax3.plot(L1_data.values, L1_data.index, color='red', linewidth=1.0)
    ax3.tick_params(axis='both', labelsize=8)
    ax3.set_xlabel('Volume Dep Ratio')
    ax3.yaxis.set_minor_locator(y3_minorlocator)
    ax3.text(0.005, height_area[1], 'e',
             verticalalignment='top',
             horizontalalignment='left',
             fontsize=15,
             )
    distancestr = 'Distance=' + str(format(min_distance, '.2f')) + 'km'
    ax3.text(horizontal[1], height_area[1] - 0.1, distancestr,
             verticalalignment='top',
             horizontalalignment='right',
             fontsize=10,
             )
    ax3.set_ylabel('Height (AGL, km)')

    ax4 = plt.subplot2grid((2, 50), (0, 19), colspan=18, rowspan=1)

    l_Dep532 = Dep532.T.loc[(Dep532.T.index < 9) & (Dep532.T.index > 0)]
    l_Dep532 = l_Dep532.iloc[::-1]
    sns.heatmap(l_Dep532, vmin=0, vmax=0.008, cmap=clrd.custom, ax=ax4, xticklabels=l_Dep532.shape[1] // 2)
    ax4.invert_yaxis()
    surface_frame = pd.DataFrame(target_surface * 100/3+3)
    ax4.set_yticks(y2_ticks)
    ax4.set_yticklabels(y_label, rotation=0)
    ax4.yaxis.set_minor_locator(y2_minorlocator)
    ax4.vlines(min_point, ymin=height_c[0], ymax=height_c[1], colors='black',
               linestyles='dashed')
    ax4.tick_params(axis='both', labelsize=8)
    ax4.plot(surface_frame.index, surface_frame.values, color='black', linewidth=2)
    ax4.fill_between(surface_frame.index, surface_frame.values.T[0], color='lightgrey')
    ax4.set_ylabel('Height (ASL, km)')
    ax4.text(2, l_Dep532.shape[0] - 3, 'c',
             verticalalignment='top',
             horizontalalignment='left',
             fontsize=15,
             )

    ax5 = plt.subplot2grid((2, 50), (1, 19), colspan=18, rowspan=1)
    l_VFM = VFM.T.loc[(VFM.T.index < 9) & (VFM.T.index > 0)]
    l_VFM = l_VFM.iloc[::-1]
    sns.heatmap(l_VFM, vmax=0.5, cmap=clrd.custom, ax=ax5, xticklabels=l_VFM.shape[1] // 2)
    ax5.invert_yaxis()
    ax5.set_yticks(y2_ticks)
    ax5.set_yticklabels(y_label, rotation=0)
    ax5.set_xlabel('Lat | Lon')
    ax5.yaxis.set_minor_locator(y2_minorlocator)
    ax5.vlines(min_point, ymin=height_c[0], ymax=height_c[1], colors='black',
               linestyles='dashed')
    ax5.plot(surface_frame.index, surface_frame.values, color='black', linewidth=2)
    ax5.fill_between(surface_frame.index, surface_frame.values.T[0], color='lightgrey')
    ax5.tick_params(axis='both', labelsize=8)
    ax5.set_ylabel('Height (ASL, km)')
    ax5.text(2, l_VFM.shape[0] - 3, 'd',
             verticalalignment='top',
             horizontalalignment='left',
             fontsize=15,
             )

    # ax[0].hlines(20, xmax=110, xmin=0, colors='black', linestyles='dashed')
    # ax[1].hlines(20, xmax=110, xmin=0, colors='black', linestyles='dashed')

    allspines_set(ax4)
    allspines_set(ax5)


def combine_proccess(date, path_SACOL, path_L1, path_vfm, path_f, time_area=None,
                     height_area=[0, 10], calibration=None, horizontal=[0.0, 0.4]):
    if not os.path.exists(path_f + '/combine/'):
        os.mkdir(path=path_f + '/combine/')
    combine_path = path_f + '/combine/' + date + '.png'
    combine_path_eps = path_f + '/combine/' + date + '.eps'
    path_L1 = path_L1
    Sacol_data = date_files_reading(date, path_SACOL)
    Sacol_data['Dp532'].values[Sacol_data['Dp532'].values < 0] = np.nan
    Sacol_data['Dp532'].values[Sacol_data['Dp532'].values > 1] = np.nan

    if date_L1_reading(date, path_L1, path_vfm) is not None:
        L1_data, min_distance, Dep532_frame, VFM_frame, target_surface, min_point = date_L1_reading(date, path_L1, path_vfm)
        Dp_height, avgdata = dep_by_height(Sacol_data['Dp532'].iloc[:, time_area[0]:time_area[1]],
                                           meantime=3, top=height_area[1], bottum=height_area[0])
        combine_plot(Sacol_data, Dep532_frame, VFM_frame, Dp_height, L1_data, target_surface, min_point, min_distance,
                     time_area, height_area, calibration, horizontal, )
        plt.savefig(combine_path, dpi=120) #png
        #plt.savefig(combine_path_eps, dpi=120) #eps
        plt.close()


def Satellite_compare(date, path_SACOL, path_L1, path_vfm, path_f, time_area=None,
                      height_area=[0, 10], calibration=None, horizontal=[0.0, 0.4]):
    if not os.path.exists(path_f + '/dep_height/'):
        os.mkdir(path=path_f + '/dep_height/')
    if not os.path.exists(path_f + '/heat_map/'):
        os.mkdir(path=path_f + '/heat_map/')
    if not os.path.exists(path_f + '/satellite/'):
        os.mkdir(path=path_f + '/satellite/')
    if not os.path.exists(path_f + '/combine/'):
        os.mkdir(path=path_f + '/combine/')

    f_path = path_f + '/dep_height/' + date
    heat_path = path_f + '/heat_map/' + date
    satellite_path = path_f + '/satellite/' + date
    combine_path = path_f + '/combine/' + date
    path_L1 = path_L1
    Sacol_data = date_files_reading(date, path_SACOL)
    Sacol_data['Dp532'].values[Sacol_data['Dp532'].values < 0] = np.nan
    Sacol_data['Dp532'].values[Sacol_data['Dp532'].values > 1] = np.nan
    L1_data, min_distance, Dep532_frame, VFM_frame, = date_L1_reading(date, path_L1, path_vfm)
    Dp_height, avgdata = dep_by_height(Sacol_data['Dp532'].iloc[:, time_area[0]:time_area[1]],
                                       meantime=3, top=height_area[1], bottum=height_area[0])
    aaa = str(avgdata)[:10]
    if calibration is not None:
        cal_Dp = Dp_height - calibration
        plot_by_height(cal_Dp, top=height_area[0], bottum=height_area[1], horizontal=horizontal)
    else:
        plot_by_height(Dp_height, top=height_area[0], bottum=height_area[1], horizontal=horizontal)

    plt.plot(L1_data.values, L1_data.index, color='red', linewidth=1.0)
    plt.title(label=min_distance)
    plt.savefig(f_path)
    plt.close()

    Radar_heat(Sacol_data, time_area, height_area)
    plt.savefig(heat_path)
    plt.close()

    f, ax = plt.subplots(nrows=2, figsize=(8, 6))
    sns.heatmap(Dep532_frame.T, vmin=0, vmax=0.5, cmap=clrd.custom, ax=ax[0])
    sns.heatmap(VFM_frame.T, vmax=7, cmap='depratio', ax=ax[1])
    # ax[0].hlines(20, xmax=110, xmin=0, colors='black', linestyles='dashed')
    # ax[1].hlines(20, xmax=110, xmin=0, colors='black', linestyles='dashed')
    allspines_set(ax[0])
    allspines_set(ax[1])
    plt.savefig(satellite_path)
    plt.close()


def Calibrate_procces(date, path, pathf, time_area=None, height_area=None, calibration=None, horizontal=[0.0, 0.4]):
    if not os.path.exists(pathf + '/dep_height/'):
        os.mkdir(path=pathf + '/dep_height/')
    if not os.path.exists(pathf + '/heat_map/'):
        os.mkdir(path=pathf + '/heat_map/')
    f_path = pathf + '/dep_height/' + date
    f_path_heat = pathf + '/heat_map/' + date  # 根据文件创立图像文件夹(可优化)

    # 文件读取，跳过文件说明，选取高度作为行名，便于画图
    Sacol_data = date_files_reading(date, path)
    Sacol_data['Dp532'].values[Sacol_data['Dp532'].values < 0] = np.nan
    Sacol_data['Dp532'].values[Sacol_data['Dp532'].values > 1] = np.nan
    l_Rdd_dic = {}

    for keys in Sacol_data:
        l_Rdd_dic[keys] = Sacol_data[keys].loc[(Sacol_data[keys].index < 10) & (Sacol_data[keys].index > 0)]
    if calibration is not None:
        l_Rdd_dic['Dp532'] = l_Rdd_dic['Dp532'] - calibration

    if (time_area is not None) & (height_area is not None):
        Dp_height, avgdata = dep_by_height(Sacol_data['Dp532'].iloc[:, time_area[0]:time_area[1]],
                                           meantime=3, top=height_area[1], bottum=height_area[0])
        aaa = str(avgdata)
        aaa = aaa[:10]
        plot_by_height(Dp_height, top=height_area[0], bottum=height_area[1], horizontal=horizontal)

        if calibration is not None:
            cal_Dp = Dp_height - calibration
            plt.plot(cal_Dp.values, cal_Dp.index, color='red', linewidth=1.0)
        plt.text(x=np.mean(horizontal), y=np.mean(height_area), s='Avg:\n' + aaa)
        plt.savefig(f_path)
        plt.close()

    Radar_heat(l_Rdd_dic, time_area, height_area)
    plt.savefig(f_path_heat)
    plt.close()


# pathf = input('Target Folder Path:')
path1 = 'E:/Datas/Atmospheric/Files Data/SACOL/NIESdat'  # 目标文件夹路径
pathfig = 'E:/Datas/Atmospheric/Files Data/SACOL/Figure/'
path_L1 = 'E:/Datas/Atmospheric/Files Data/SACOL/L1_data/'
path_vfm = 'E:/Datas/Atmospheric/Files Data/SACOL/VFM_data/'

if not os.path.exists(pathfig):  # 文件夹创建，用于保存图片，若存在则在不创建
    os.mkdir(path=pathfig)

cal_dic1 = {
    '20181217': [[120, 132], [2, 2.5]],
    '20190116': [[108, 120], [2, 3]],
    '20190704': [[6, 18], [2.5, 3]],
    '20190710': [[12, 24], [1.5, 2]],
}
cal_dic2 = {
    '20191026': [[84, 96], [2.5, 3]],
    '20191028': [[36, 48], [2.5, 3.5]],
}
cal_dic3 = {
    '20191113': [[120, 132], [2, 2.5]],
    '20191209': [[6, 18], [2.5, 3]],
    '20191222': [[6, 18], [2, 2.5]],
    '20200309': [[72, 84], [1.8, 2.2]]
}
cal_dic4 = {
    '20200430': [[108, 120], [4.2, 5.5]],
    '20200501': [[18, 30], [3.9, 4.3]],
}
cal_dic5 = {
    '20200617': [[84, 96], [3.4, 4]],
    '20200801': [[90, 102], [3.1, 3.5]],
    '20200918': [[54, 66], [3.0, 3.5]],
}
cal_main_dic = {
    '1': cal_dic1,
    '2': cal_dic2,
    '3': cal_dic3,
    '4': cal_dic4,
    '5': cal_dic5,
}

process_list = ['1', '2', '3', '4', '5']

satel_dic1 = {
    '20181116': [[111, 117], [0, 7]],
    '20190125': [[111, 117], [0, 8]],
    '20190418': [[112, 118], [0, 8]],
    '20190501': [[112, 118], [0, 5]],
    '20190805': [[112, 118], [0, 5]],
    '20190314': [[37, 43], [0, 8]],
    '20181221': [[37, 43], [0, 4]],
}
satel_dic2 = {
    '20181116': [[111, 117], [0, 9]],
}
satel_dic3 = {
    '20191113': [[37, 43], [0, 9]],
    '20191222': [[37, 43], [0, 9]],
    '20191209': [[37, 43], [0, 9]],
    '20200104': [[37, 43], [0, 9]],
    '20200130': [[37, 43], [0, 9]],
    '20200212': [[37, 43], [0, 9]],
    '20200225': [[37, 43], [0, 9]],
    '20200309': [[37, 43], [0, 9]],
}
satel_dic4 = {
    '20200430': [[37, 43], [0, 9]],
    '20200513': [[37, 43], [0, 9]],
}
satel_dic5 = {
    '20200612': [[112, 118], [0, 9]],
    '20200625': [[112, 118], [0, 9]],
}

satel_main_dic = {
    '1': satel_dic1,
    '2': satel_dic2,
    '3': satel_dic3,
    '4': satel_dic4,
    '5': satel_dic5,
}
compare_list = ['1', '3', '4', '5']

cal_dic = {
    '1': 0.0009543304571500684,
    '2': 0.005401334679040032,
    '3': 0.007466093533295523,
    '4': 0.026124242527262673,
    '5': 0.006568091719789208,
}

os.chdir(path1)
all_file_list = os.listdir()
for file in all_file_list:
    path_all = pathfig + 'all/'
    if not os.path.exists(path_all):
        os.mkdir(path=path_all)
    if file[-4:] == '.dat':
        date = file[16:24]
        i_date = int(date)
        t_date = date[0:4] + '-' + date[4:6] + '-' + date[6:8]
        os.chdir(path_L1)
        L1_list = os.listdir()
        for files in L1_list:
            fname = re.match('^CAL_LID_L1-Standard-V4-10\.' + t_date + '.*\.hdf$', files)
            if fname is not None:
                print(files[45:47])
                t_area = [37, 43]
                if files[45:47] =='ZD':
                    t_area = [37, 43]

                elif files[45:47] =='ZN':
                    t_area = [112, 118]

                if (i_date >= 20181100) & (i_date < 20191023):
                    date_case = '1'
                elif (i_date >= 20191023) & (i_date < 20191029):
                    date_case = '2'
                elif (i_date >= 20191029) & (i_date < 20200427):
                    date_case = '3'
                elif (i_date >= 20200427) & (i_date < 20200516):
                    date_case = '4'
                elif (i_date >= 20200516) & (i_date < 20220101):
                    date_case = '5'
                else:
                    date_case = '0'

                path_plot_dir = path_all + date_case
                if not os.path.exists(path_plot_dir):
                    os.mkdir(path=path_plot_dir)
                print(date_case)
                print(i_date)
                combine_proccess(date, path1, path_L1, path_vfm, path_plot_dir, time_area=t_area,
                                 height_area=[0, 8], calibration=cal_dic[date_case], horizontal=[0.0, 0.4])


'''
for num in process_list:
    path_plot_dir = pathfig + num
    if not os.path.exists(path_plot_dir):
        os.mkdir(path=path_plot_dir)
    cal_list = []
    for key in cal_main_dic[num]:
        Calibrate_procces(key, path1, path_plot_dir, time_area=cal_main_dic[num][key][0],
                          height_area=cal_main_dic[num][key][1], horizontal=[0, 0.02])
        avg_dp, dp_height = target_average_dp(key, path1, time_area=cal_main_dic[num][key][0],
                                              height_area=cal_main_dic[num][key][1])
        cal_list.append(avg_dp - 0.0044)
    cal_dic[num] = np.min(cal_list)

    path_plot_dir = pathfig + num + '_all_height'
    if not os.path.exists(path_plot_dir):
        os.mkdir(path=path_plot_dir)

    for key in cal_main_dic[num]:
        Calibrate_procces(key, path1, path_plot_dir, time_area=cal_main_dic[num][key][0],
                          height_area=[0, 5], calibration=cal_dic[num], horizontal=[0, 0.1])


'''

for num in compare_list:
    path_plot_dir = pathfig + num + '_satellite'
    if not os.path.exists(path_plot_dir):
        os.mkdir(path=path_plot_dir)

    for key in satel_main_dic[num]:
        combine_proccess(key, path1, path_L1, path_vfm, path_plot_dir, time_area=satel_main_dic[num][key][0],
                         height_area=satel_main_dic[num][key][1], calibration=None, horizontal=[0.0, 0.4])

'''

print(cal_dic)

Dp_height, avgdata = dep_by_height(Rddata_dic['Dp532'].loc['12:00':'17:00'], meantime=1)
print(avgdata)

plot_by_height(Dp_height)

plt.savefig(f_path)
plt.close()

l_Rdd_dic['Dp532'].values[l_Rdd_dic['Dp532'].values < 0] = 0
l_Rdd_dic['Dp532'].values[l_Rdd_dic['Dp532'].values > 1] = 1
print(l_Rdd_dic['Dp532'])
plot_by_height(l_Rdd_dic['Dp532'].iloc[:, 80])

'''
