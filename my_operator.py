# M always means matrix
import numpy as np
import pandas as pd
import os
# import re
# import string
# import gzip
# from numba import vectorize
import numpy.linalg as la
import copy

def vis_camelyon(data,tile = 95, cut = 5):
    _th_ = np.percentile(data[np.where(data!=0)],tile)
    data = (data - _th_)/(np.max(data) - _th_)
    data[data<=0] = 0
    cut = 5
    # if cut == 5:
    #     _th_1 = np.percentile(data[np.where(data!=0)],20)
    #     _th_2 = np.percentile(data[np.where(data!=0)],40)
    #     _th_3 = np.percentile(data[np.where(data!=0)],60)
    #     _th_4 = np.percentile(data[np.where(data!=0)],80)
    #     # feat_att[(feat_att)>0 & (feat_att<= _th_1)] = 0.2
    #     # feat_att[(feat_att)>_th_4] = 1
    #     data[data>_th_4] = 1
    #     data[(data>_th_3) & (data<= _th_4)] = 0.8
    #     data[(data>_th_2) & (data<= _th_3)] = 0.6
    #     data[(data>_th_1) & (data<= _th_2)] = 0.4
    #     data[(data>0) & (data<= _th_1)] = 0.2
    # elif cut == 10:
    map_values = np.arange(1/cut,1+1/cut,1/cut) #np.flip() ##array([0.2, 0.4, 0.6, 0.8, 1. ])
    quantiles = np.arange(0,100,100/cut)
    for i in range(cut):
        locals()["_th_" + str(i)] = np.percentile(data[np.where(data!=0)],quantiles[i])
    locals()["_th_" + str(cut)] = np.max(data)
    for i in np.arange(cut-1,-1,-1): ### 4,3,2,1,0
        data[(data>eval("_th_" + str(i))) & (data<= eval("_th_" + str(i+1)))] = map_values[i]
    return data

def compute_total_var_ell2(img):      
    tv_h = (np.power((img[1:,:] - img[:-1,:]),2)).sum()
    tv_w = (np.power((img[:,1:] - img[:,:-1]),2)).sum()    
    return (tv_h + tv_w)

def shrinkage_operator(X, tu):
    return np.sign(X) * np.maximum((np.abs(X) - tu), np.zeros(X.shape))

def mask_operator(X, tu):
    X[np.abs(X) <= tu] = 0
    return X

def Vec(M):
    '''return shape is m*n,1 '''
    return M.reshape(-1,1)
def Vec_inv(M,m,n,d=1):
    # stack direction: axis = 1
    if d == 1:
        return M.reshape(m,n)
    else:
        return M.reshape(m,n,d)
def Rearrange(C,p1,d1,p2,d2,p3=1,d3=1):
    flag = C.shape
    if len(flag) == 2:
        m,n = C.shape
        RC = []
        assert m == p1*d1 and n == p2*d2, "Matrix dimension wrong !!!!!"
        for i in range(p1):
            for j in range(p2):
                Cij = C[d1*i:d1*(i+1),d2*j:d2*(j+1)]
                RC.append(Vec(Cij))
        return np.concatenate(RC,axis = 1).T
    elif len(flag) == 3:
        m,n,d = C.shape
        RC = []
        assert m == p1*d1 and n == p2*d2 and d == p3*d3, "Tensor dimension wrong !!!!"
        for i in range(p1):
            for j in range(p2):
                for k in range(p3):
                    Cij = C[d1*i:d1*(i+1),d2*j:d2*(j+1),d3*k:d3*(k+1)]
                    RC.append(Vec(Cij))
        return np.concatenate(RC,axis = 1).T

def R_inv_guang(RM, blockshape, idctshape):
    m, n = RM.shape
    d1, d2 = blockshape
    p1, p2 = idctshape
    assert m == p1 * p2 and n == d1 * d2, "Dimension wrong"
    M = np.zeros([d1 * p1, d2 * p2])
    for i in range(m):
        Block = Vec_inv(RM[i, :], blockshape[0],blockshape[1])
        ith = i // p2  # quotient
        jth = i % p2  # remainder
        M[d1*ith: d1*(ith+1), d2*jth: d2*(jth+1)] = Block

    return M

def R_opt_pro(A, idctshape):
    m, n = A.shape
    p1, p2 = idctshape
    assert m % p1 == 0 and n % p2 == 0, "Dimension wrong"
    d1, d2 = m // p1, n // p2
    strides = A.itemsize * np.array([p2*d2*d1, d2, p2*d2, 1])
    A_blocked = np.lib.stride_tricks.as_strided(A, shape=(p1, p2, d1, d2), strides=strides)
    # RA = A_blocked.reshape(-1, d1*d2)
    return A_blocked

def R_inv(RC,p1,d1,p2,d2,p3=1,d3=1):
    if p3 ==1 and d3 ==1:
        p1p2,d1d2 = RC.shape
        C = np.zeros([p1*d1,p2*d2])
        bb = []
    #     print("d1: ",d1)
    #     print("d2: ",d2)

        for i in range(p1p2):

            Block = Vec_inv(RC[i,:],d1,d2)
            ith = i // p2 # quotient
            jth = i % p2  # remainder
            C[d1*ith:d1*(ith+1),d2*jth:d2*(jth+1)] = Block
            bb.append(Block)

        return C,bb
    else:
        p1p2p3,d1d2d3 = RC.shape
        print("RC shape: ",RC.shape)
        C = np.zeros([p1*d1,p2*d2,p3*d3])
        p2p3 = p2*p3 
        bb = []
        print("p2p3",p2*p3)
        for i in range(p1p2p3):
            Block = Vec_inv(RC[i,:],d1,d2,d3)
            # print("i: ",i)
            ith = i //  p2p3 ## quotient
            reminder = i % p2p3  # reminder
            jth = reminder // p3
            kth = reminder % p3
            C[d1*ith:d1*(ith+1),d2*jth:d2*(jth+1),d3*kth:d3*(kth+1)] = Block
            bb.append(Block)
        return C,bb

# for plot gray image 
def convert_to_gray(img):
    m,n = img.shape
    for i in range(m):
        for j in range(n):
            if img[i,j] == 0:
                img[i,j] = 1
            elif img[i,j] ==1 :
                img[i,j] = 0
    return img

# bivariate Haar wavelet basis function
def bi_haar(x):
    if x>= -1 and x < 0:
        x = 1
    elif x >= 0 and x <1 :
        x = -1
    else:
        x = 0
    return x

def RELU(x):
    return np.maximum(0,x)

def sigmoid(x):
    return (1/(1+np.exp(-0.1*x))) * 20

def func_kron_ab(a_hat,b_hat,R,p1,p2,d1,d2,p3=1,d3 = 1):
    Ra_hat = a_hat.reshape(R,-1)
    Rb_hat = b_hat.reshape(R,-1)
    A = []
    B = []
    kron_ab = []
    if p3 == 1 and d3 == 1:
        for i in range(1,R+1):
            locals()['a_hat' + str(i)] = Ra_hat[i-1,:].reshape(-1,1)
            locals()['b_hat' + str(i)] = Rb_hat[i-1,:].reshape(-1,1)
            A.append(Vec_inv(eval('a_hat' + str(i)),p1,p2))
            B.append(Vec_inv(eval('b_hat' + str(i)),d1,d2))

        kron_ab = [np.kron(A[i],B[i]) for i in range(R)]
        beta_hat = sum(kron_ab)
        kron_ab.append(beta_hat)
        
    else:
        Ra_hat = a_hat.reshape(R,-1)
        Rb_hat = b_hat.reshape(R,-1)
        A = []
        B = []
        kron_ab = []
        for i in range(1,R+1):
            locals()['a_hat' + str(i)] = Ra_hat[i-1,:].reshape(-1,1)
            locals()['b_hat' + str(i)] = Rb_hat[i-1,:].reshape(-1,1)
            A.append(Vec_inv(eval('a_hat' + str(i)),p1,p2,p3))
            B.append(Vec_inv(eval('b_hat' + str(i)),d1,d2,d3))

        kron_ab = [np.kron(A[i],B[i]) for i in range(R)]
        beta_hat = sum(kron_ab)
        kron_ab.append(beta_hat)

    return A,B,kron_ab

def fun_th(X):
    ## set  non-zero entries in matrix X to 1
    return np.where(np.abs(X) == 0 ,X,1)

def fun_normalization(data):
    data = np.abs(data)
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def fun_average(data,num = 7):
    ## data is a  list
    new_data = []
    tmp = []
    for i in range(1,len(data)+1,1):
        tmp.append(data[i])
        if i%num ==0:
            new_data.append(np.mean(np.array(tmp),axis = 0))
            tmp = []
    return new_data


def fun_preprocess(Xi):
    fro = la.norm(Xi)
    mu = np.mean(Xi)
    std = np.std(Xi)
    return (Xi-mu)/std

def min_max_norm(Xi):
    _max = np.max(Xi)
    _min = np.min(Xi)
    return (Xi-_min)/(_max-_min)
def thresh_normalization(tmp,tile = 99):
    data = copy.deepcopy(tmp)
    mark = np.percentile(data[np.where(data!=0)],tile)
#     print(mark)
    data = np.where(data <= mark ,data,mark)
    return data

def Gaussian_lize(C, mu=1, sigma=1):
    idx = np.where(C != 0)
    C[idx[0], idx[1]] = np.random.normal(mu, sigma, len(idx[0]))
    return C

from nilearn.plotting import plot_stat_map, show
from nilearn.input_data import NiftiMasker
from nilearn.image import new_img_like
from nilearn.image import get_data
import nibabel as nib
from nibabel import nifti1
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
def ni_show(beta_hat,threshold = 0,cut_coords = None,save_name = None):
    ss = 3
    file_name = "1497923_20252_2_0"  ##1496176_20252_2_0
    img = nib.load("T1_brain_to_MNI.nii.gz")
    re_th = resize(beta_hat,(144,184,144))
    canvas = np.zeros(list(img.dataobj.shape))
    canvas[7+4*ss:-7-4*ss,5+4*ss:-5-4*ss,4*ss-4:-18-4*ss] = re_th
    new_image = nib.Nifti1Image(fun_normalization(canvas), img.affine)
    fig = plt.figure(figsize=(9, 3), facecolor='w')
    # background = new_img_like(img,new_image)
    display = plot_stat_map(stat_map_img = new_image,bg_img = img,
                            threshold=threshold,
                            colorbar = True,
                            draw_cross = False,
                            cut_coords= cut_coords,
                            black_bg = False,
                            output_file = save_name,
                            figure=fig)
    show()
def fun_pertile(data,tile = 90):
    data = np.abs(data)
    mark = np.percentile(data[np.where(data!=0)],tile)
    print(mark)
    data = np.where(data < mark ,data,mark)
    _range = mark - np.min(data)
    return (data - np.min(data)) / (_range)