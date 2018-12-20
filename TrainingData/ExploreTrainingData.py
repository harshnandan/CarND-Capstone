import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import random


fname = 'Training2018112363121'

fid = open(fname,'rb')
data = pickle.load(fid)
fid.close()

# Get information about data size
data_size = data['Xtrain'].shape[0]
print('Total number of data %d'%(data_size,))

# Randomly pick data and plot
data_idx = random.sample(range(0,data_size), 8)
plt.figure()
for seq,idx in enumerate(data_idx):
    orig_img = data['Xtrain'][idx]
    tmp_img = np.zeros_like(orig_img)

    # Flip second and third color element of orig_img
    tmp_img[:,:,0] = orig_img[:,:,2]
    tmp_img[:,:,1] = orig_img[:,:,1]
    tmp_img[:,:,2] = orig_img[:,:,0]

    plt.subplot(2,4,seq+1)
    plt.imshow(tmp_img)

    # Check state of traffic-light
    state = data['YLabel'][idx]
    if state == 0:
        tf_color = 'red'
    elif state == 1:
        tf_color = 'yellow'
    elif state == 2:
        tf_color = 'green'
    else:
        tf_color = 'unknown'

    # Check distance from ego to traffic light
    dist = data['EgoTrafficDist'][idx]

    title_str = 'Color : %s , Dist : %d m'%(tf_color, dist)
    plt.title(title_str)
    plt.axis('off')

# Second plot
data_idx = random.sample(range(0,data_size), 8)
plt.figure()
for seq,idx in enumerate(data_idx):
    orig_img = data['Xtrain'][idx]
    tmp_img = np.zeros_like(orig_img)

    # Flip second and third color element of orig_img
    tmp_img[:,:,0] = orig_img[:,:,2]
    tmp_img[:,:,1] = orig_img[:,:,1]
    tmp_img[:,:,2] = orig_img[:,:,0]

    plt.subplot(2,4,seq+1)
    plt.imshow(tmp_img)

    # Check state of traffic-light
    state = data['YLabel'][idx]
    if state == 0:
        tf_color = 'red'
    elif state == 1:
        tf_color = 'yellow'
    elif state == 2:
        tf_color = 'green'
    else:
        tf_color = 'unknown'

    # Check distance from ego to traffic light
    dist = data['EgoTrafficDist'][idx]

    title_str = 'Color : %s , Dist : %d m'%(tf_color, dist)
    plt.title(title_str)
    plt.axis('off')


# Show all plots
plt.show()

