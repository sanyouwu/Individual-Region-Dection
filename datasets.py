from sklearn.metrics import confusion_matrix
import numpy as np
import copy
import scipy.linalg as la
from skimage import draw


name = "MUSK1"
file = arff.loadarff(os.path.join(img_dir,"nweka-musk1-10-1tra.arff"))
# img_dir = r"D:\Codes\IndividualizedRegionSelectionMIL\raw_data\ColonCancer\Detection\img1"
from scipy.io import loadmat
from PIL import Image
k = 18
X, Y, groudTruth = [], [], []
for k in range(1,101):
    # print(k)
    img_dir = r"D:\Codes\IndividualizedRegionSelectionMIL\raw_data\ColonCancer\Classification\img"+str(k)
    _info = loadmat(os.path.join(img_dir,"img"+str(k)+"_epithelial.mat"))
    _detection = np.round(_info["detection"],0)
    data = np.array(Image.open(os.path.join(img_dir,"img"+str(k)+".bmp")).convert('L'))
    mask = np.zeros((500,500))
    for i in range(_detection.shape[0]):
        x,y = _detection[i,:]
        mask[int(x)-10:int(x)+10,int(y)-10:int(y)+10] = 1
    if len(_detection) == 0:
        Y.append(0)
    else:
        Y.append(1)
    X.append(data)
    groudTruth.append(mask.T)



