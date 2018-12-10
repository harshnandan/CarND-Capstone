import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import random
import cv2


fname = 'Training20181124175713'

fid = open(fname,'rb')
data = pickle.load(fid)
fid.close()

green_count = 114

# Get information about data size
data_size = data['Xtrain'].shape[0]
print('Total number of data %d'%(data_size,))


# Looping over data
for idx in range(data_size):
    label = data['YLabel'][idx]

    if label == 2:
        # Red-light
        green_count = green_count + 1
        cv2.imwrite('green--'+str(green_count)+'.jpg', data['Xtrain'][idx])
