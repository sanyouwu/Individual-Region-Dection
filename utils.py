from sklearn.metrics import confusion_matrix
import numpy as np
import copy
import scipy.linalg as la
from skimage import draw

from scipy.interpolate import griddata
import torch
import torch.nn as nn

def recover_P(p1,p2, y_hat,idx_stay):
    # coords_scale = np.round(,0).astype(int)
    P = np.zeros(p1 * p2)
    P[idx_stay] = y_hat.flatten()
    return P.reshape(p1, p2)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def plot_contour(x,y,z,resolution = 50,contour_method='linear'):
    resolution = str(resolution)+'j'
    X,Y = np.mgrid[min(x):max(x):complex(resolution),   min(y):max(y):complex(resolution)]
    points = [[a,b] for a,b in zip(x,y)]
    Z = griddata(points, z, (X, Y), method=contour_method,)# fill_value= 255
    return X,Y,Z

def min_max_normalization(x):
    _min = np.min(x)
    _max = np.max(x)
    return (x-_min)/_max


### for breast cancer
def check_white(x, threshld = 240):
    img_type = len(x.shape)
    if img_type == 3:
        # rgb
        x = np.mean(x, axis = 2)

    d1,d2 = x.shape
    d = d1 * d2
    num_white = np.where(x >= threshld)[0].shape[0]
    _cut_rate = 0.75
      ## 0.75

    if num_white/d >= _cut_rate:
        return 1

    else:
        return 0

def remove_white(patch_all,threshld = 240):
    ''' img is path all
        e.g., tmp with dimension 672 x 32 x 32 x 3  = p x d_1 x d_2 

        remove some white patch and record the removed index

        return the new_pathch_list and removed_index
    '''
    P = len(patch_all)
    # new_patch_list = [check_white i in range(P)]
    idx_removed = []
    idx_stay = []
    for i in range(P):
        x = patch_all[i]
        flag = check_white(x,threshld) ## flag = 1 means white image, should be removed later
        if flag:
            idx_removed.append(i)
        else:
            idx_stay.append(i)
    img_after_rm = np.delete(patch_all, idx_removed, axis = 0)
    return img_after_rm, idx_stay


def recover_white(img_after_rm, idx_stay, P):
    '''
    this input is the output of remove_white function
    '''
    # img = 0
    P_removed, d1,d2, channel = img_after_rm.shape
    patch_all = np.ones((P, d1,d2, channel))
    patch_all[idx_stay] = img_after_rm
    return patch_all

def block_to_img(T_blocked,p1,p2):
    P,d1,d2,channel = T_blocked.shape
    D1 = p1 * d1
    D2 = p2 * d2
    out = T_blocked.reshape(p1,p2,d1,d2,3).transpose(0,2,1,3,4).reshape(D1,D2,3)
    return out.astype(int)
### measurements
def patchwise_auc(data):
    auc = 0

    return auc

def cls_measure(truth,pred):
    TN, FP, FN, TP = confusion_matrix(truth,pred).flatten()
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP) 
    FPR = FP/(FP+TN)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    return TPR, TNR, ACC

def R_inv_T(RT, Adim, Bdim):
    ''' Inverse R operator for tensor: (p1*p2*p3, d1*d2*d3) to (p1*d1, p2*d2, p3*d3) '''
    P, D = RT.shape
    p1, p2, p3 = Adim
    d1, d2, d3 = Bdim
    assert P == p1 * p2 * p3 and D == d1 * d2 * d3, 'Dimension wrong!'
    slices = []
    fibers = []
    for i in range(P):
        fiber = RT[i, ].reshape(Bdim)
        fibers.append(fiber)
        if len(fibers) == p2:
            slice = np.concatenate(fibers, axis=1)
            slices.append(slice)
            fibers = []
    T = np.concatenate(slices, axis=0)
    return T

def cut_T_pro(T, Adim, img_type = "gray"):
    ''' R operator for matrix/tensor: (p1*d1, p2*d2, p3*d3) to (p1*p2*p3, d1*d2*d3) '''
    if len(Adim) == 2:
        N1, N2 = list(T.shape)[0], list(T.shape)[1]
        p1, p2 = Adim
        assert N1 % p1 == 0 and N2 % p2 == 0, "Dimension wrong"
        d1, d2 = N1 // p1, N2 // p2
        if img_type == "rgb":
            strides = T.itemsize * np.array([p2*d2*d1*3, d2*3, p2*d2*3, 3,1])
            T_blocked = np.lib.stride_tricks.as_strided(T, shape=(p1, p2, d1, d2,3), strides=strides)
        else:
            strides = T.itemsize * np.array([p2*d2*d1, d2, p2*d2, 1])
            T_blocked = np.lib.stride_tricks.as_strided(T, shape=(p1, p2, d1, d2), strides=strides)
    else:
        N1, N2, N3 = T.shape
        p1, p2, p3 = Adim
        assert N1 % p1 == 0 and N2 % p2 == 0 and N3 % p3 == 0, "Dimension wrong"
        d1, d2, d3 = N1 // p1, N2 // p2, N3 // p3
        strides = T.itemsize * np.array([N2 * N3 * d1, N3 * d2, d3, N2 * N3, N3, 1])  # 大层，大行，大列，小层，小行，小列
        T_blocked = np.lib.stride_tricks.as_strided(T, shape=(p1, p2, p3, d1, d2, d3), strides=strides)
    
    return T_blocked

def shift_circle(img,x0,y0,r,n,shift_inten = 6, signal_inten = 1):
    # center is (x0,y0)
    # intensity 
    # np.random.seed(539)
    D1,D2 = img.shape
    canvas = np.zeros((D1,D2))
    img_list = []
    # np.random.seed(seed)
    shift_x = np.random.uniform(low=-1, high= 1, size= n) * shift_inten
    shift_y = np.random.uniform(low=-1, high= 1, size= n) * shift_inten
    ins_label_list = []
    for i in range(n):
        tmp = circle(img,x0 + shift_x[i],y0 + shift_y[i],r,signal_inten)
        ins_tmp = circle(canvas,x0 + shift_x[i],y0 + shift_y[i],r,1)
        img_list.append(tmp)
        ins_label_list.append(ins_tmp)
    return img_list,ins_label_list

# def shift_circle(img,x0,y0,r,n,shift_inten = 6, signal_inten = 1):
#     # center is (x0,y0)
#     # intensity 
#     np.random.seed(539)
#     D1,D2 = img.shape
#     canvas = np.zeros((D1,D2))
#     img_list = []
#     shift_x = np.random.uniform(low=-1, high= 1, size= n) * shift_inten
#     shift_y = np.random.uniform(low=-1, high= 1, size= n) * shift_inten
#     ins_label_list = []
#     for i in range(n):
#         tmp = circle(img,x0 + shift_x[i],y0 + shift_y[i],r,signal_inten)
#         ins_tmp = circle(canvas,x0 + shift_x[i],y0 + shift_y[i],r,1)
#         img_list.append(tmp)
#         ins_label_list.append(ins_tmp)
#     return img_list,ins_label_list

# def circle(img,x0,y0,r):
#     temp = copy.deepcopy(img)
#     m,n = temp.shape
#     for i in range(m):
#         for j in range(n):
#             dist = np.round(la.norm(np.array([i-x0, j-y0]),2))
#             if dist <= r:
#                 temp[i,j] = 1
#     return temp

from skimage.draw import disk
from skimage import draw
def circle(img,x0,y0,r,signal_inten = 1):
    arr = copy.deepcopy(img)
    # rr, cc = draw.disk((x0, y0), radius=r, shape=arr.shape)
    rr, cc = disk((x0, y0), radius=r, shape=arr.shape)
    arr[rr, cc] = signal_inten
    return arr


def np_find(arr,_min,_max):
    arr = np.array(arr)
    pos_min  = arr >= _min
    pos_max = arr <= _max
    pos_rst = pos_min & pos_max
    return np.where(pos_rst == True)

def min_max_norm(Xi):
    _max = np.max(Xi)
    _min = np.min(Xi)
    return (Xi-_min)/(_max-_min)

# def drop_noninfo(data):
#     '''
#     data: P x d1 x d2
#     '''
#     P,d1,d2 = data.shape
#     for _slice in data:
        
def TV(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X-mean)/std
    m,n = X.shape
    gX = np.zeros([m,n])
    for j in range(m):
        for k in range(n):
            if j<= m-2 and k <= n-2:
                g = np.asarray([X[j+1,k]-X[j,k],X[j,k+1]-X[j,k]])
            if j==m-1 and k <= n-2:
                g = np.asarray([0,X[j,k+1]-X[j,k]])
            if j<=m-2 and k == n-1:
                g = np.asarray([X[j+1,k]-X[j,k],0])              
            gX[j,k] = la.norm(g,2)**2
#             gX[j,k] = np.sum(np.abs(g))
    return gX


from xml.dom import minidom
import xml.etree.ElementTree as ET
import cv2
def initXML(xml_path):
    def _createContour(coord_list):
        return np.array([[[int(float(coord.attributes['X'].value)), 
                               int(float(coord.attributes['Y'].value))]] for coord in coord_list], dtype = 'int32')

    xmldoc = minidom.parse(xml_path)
    annotations = [anno.getElementsByTagName('Coordinate') for anno in xmldoc.getElementsByTagName('Annotation')]
    contours_tumor  = [_createContour(coord_list) for coord_list in annotations]
    contours_tumor = sorted(contours_tumor, key=cv2.contourArea, reverse=True)
    return contours_tumor


### for visualization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
def draw(ylabel, ylim, xticklabels, title, string, data, errorbar, ytick,width, xlabel, rotation=0):
    patterns = [".", "x", "|", "o", "*", "O", "/", "\\", "+", "-", "H"]
    color = ['yellow', 'lightcoral', 'deepskyblue', 'violet', 'Grey', 'red', 'magenta', 'cyan', 'brown', 'white', 'brown', 'teal', 'green']
    name = string #+ '_' + title + '_' + ylabel

    opacity = 0.26
    num = len(data)
    ind = np.arange(num)  # the x locations for the groups
    # width = 0.55  # the width of the bars
    fig, ax = plt.subplots(figsize=(8, 6))
    # rects1 = ax.bar(ind, data, width, color=color, alpha=1, hatch=patterns)# edgecolor='black',
    # y_pos = [0, 2, 4, 6]
    for i in range(num):
        ax.bar(ind[i], data[i], width, color=color[i], edgecolor='black', hatch=patterns[i], yerr=errorbar[i], capsize=7)
    ax.set_title(title, fontsize=27)
    ttl = ax.title
    ttl.set_position([.5, 1.02])

    ax.set_xticks(ind)
    ax.set_xticklabels(xticklabels, rotation=rotation, fontsize=21)
    # ax.set_xlabel(title, weight='bold', fontsize=27)

    plt.ylim(ylim[0], ylim[1])
    ax.yaxis.set_ticks(ytick)
    plt.yticks(fontsize=21) #weight='bold',
    ax.set_ylabel(ylabel, fontsize=30)

    ax.yaxis.grid(ls='dashed')
    # plt.grid(ls='dashed')
    plt.tight_layout(pad=0)
    if name is not None:
        # plt.savefig(name + '.pdf')
        plt.savefig(name + ".pdf", dpi = 300, bbox_inches = "tight", pad_inches = 0)
        plt.savefig(name + '.png', dpi=300)
    plt.show()