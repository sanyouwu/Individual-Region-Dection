"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 0903, last modified in 2021 0511.
"""

import numpy as np
import scipy.io as scio
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_file(para_path):
    """
    Load file.
    :param
        para_file_name:
            The path of the given file.
    :return
        The data.
    """
    temp_type = para_path.split('.')[-1]

    if temp_type == 'mat':
        ret_data = scio.loadmat(para_path)
        return ret_data['data']
    else:
        with open(para_path) as temp_fd:
            ret_data = temp_fd.readlines()

        return ret_data


def print_progress_bar(para_idx, para_len):
    """
    Print the progress bar.
    :param
        para_idx:
            The current index.
        para_len:
            The loop length.
    """
    print('\r' + '▇' * int(para_idx // (para_len / 50)) + str(np.ceil((para_idx + 1) * 100 / para_len)) + '%', end='')


def mnist_bag_loader(train, mnist_path=None):
    """"""
    if mnist_path is None:
        mnist_path = "../../Data"
    return DataLoader(datasets.MNIST(mnist_path,
                                     train=train,
                                     download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])),
                      batch_size=1,
                      shuffle=False)


def get_k_cross_validation_index(num_x, k=10):
    """
    The get function.
    """
    rand_idx = np.random.permutation(num_x)
    temp_fold = int(np.floor(num_x / k))
    ret_tr_idx = []
    ret_te_idx = []
    for i in range(k):
        temp_tr_idx = rand_idx[0: i * temp_fold].tolist()
        temp_tr_idx.extend(rand_idx[(i + 1) * temp_fold:])
        ret_tr_idx.append(temp_tr_idx)
        ret_te_idx.append(rand_idx[i * temp_fold: (i + 1) * temp_fold].tolist())
    return ret_tr_idx, ret_te_idx
