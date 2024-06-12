import h5py
import hdf5storage
import scipy.io as scio
import numpy as np
def load_data(path):
    file=h5py.File(path)
    imagetest = file['I_te'][:].astype('float').T
    imagetrain = file['I_tr'][:].astype('float').T
    labelstest = file['L_te'][:].T
    labelstrain =file['L_tr'][:].T
    texttest = file['T_te'][:].T
    texttrain =file['T_tr'][:].T


    return imagetest, imagetrain, labelstest, labelstrain, texttest, texttrain


def load_pretrain_model(path):
    return scio.loadmat(path)


if __name__ == '__main__':
    a = {'s': [12, 33, 44],
         's': 0.111}
    import os
    with open('result.txt', 'w') as f:
        for k, v in a.items():
            f.write(k, v)(base)