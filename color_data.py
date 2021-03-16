import numpy as np
import matplotlib.colors as clrs

x_range = np.linspace(0, 255, 256)
r_f = np.interp(x_range, [0, 1, 50, 150, 200, 255], [255, 255, 0, 10, 255, 255])
g_f = np.interp(x_range, [0, 1, 50, 100, 200, 255], [255, 255, 0, 255, 255, 0])
b_f = np.interp(x_range, [0, 100, 150, 255], [255, 255, 10, 0])
_custom_data = (np.concatenate([[r_f], [g_f], [b_f]]).T)/255

r_f1=[  0,  1, 40,255,253,255,253,253,170,255]
g_f1=[  0,176,211,255,172,  0,  1,211, 83,255]
b_f1=[255,250,  1,  0,  1,  0,253,253,255,255]

_dep_ratio_color = (np.concatenate([[r_f1], [g_f1], [b_f1]]).T)/255

r_VFM = [0, 0, 255, 250, 0, 211, 0]
g_VFM = [38, 220, 160, 255, 255, 211, 0]
b_VFM = [255, 255, 0, 0, 110, 211, 0]

_VFM_color = (np.concatenate([[r_VFM], [g_VFM], [b_VFM]]).T)/255

custom = clrs.ListedColormap(_custom_data)
dep = clrs.ListedColormap(_dep_ratio_color)
VFM = clrs.ListedColormap(_VFM_color)