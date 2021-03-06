import numpy as np
import random
import cPickle as pickle
import matplotlib.pyplot as plt

from keras.layers.core import Dense, Flatten, Lambda, Dropout
from keras.layers.convolutional import Cropping2D, Conv2D
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.backend import tf as ktf
import keras.losses
import glob


def exploration_plot(data):
    # Randomly pick data and plot
    data_size = data['Xtrain'].shape[0]
    data_idx = random.sample(range(0, data_size), 20)
    plt.figure()
    for seq, idx in enumerate(data_idx):
        orig_img = data['Xtrain'][idx]
        tmp_img = np.zeros_like(orig_img)

        # Flip second and third color element of orig_img
        tmp_img[:, :, 0] = orig_img[:, :, 2]
        tmp_img[:, :, 1] = orig_img[:, :, 1]
        tmp_img[:, :, 2] = orig_img[:, :, 0]

        plt.subplot(4, 5, seq + 1)
        plt.imshow(tmp_img)

        # Check state of traffic-light
        state = data['YLabel'][idx]
        if (state == 0 ):
            tf_color = 'red / yellow'
        elif state == 1:
            tf_color = 'green'
        else:
            tf_color = 'unknown'

        # Check distance from ego to traffic light
        dist = data['EgoTrafficDist'][idx]

        title_str = 'Color : %s , Dist : %d m' % (tf_color, dist)
        plt.title(title_str)
        plt.axis('off')
    plt.show()

def split_data(whole_data, split_frac):
    data_len = whole_data['Xtrain'].shape[0]

    # Shuffle data
    shuffle_idx = random.sample(range(0,data_len),data_len)
    whole_data['Xtrain'] = whole_data['Xtrain'][shuffle_idx,:,:,:]
    whole_data['YLabel'] = [whole_data['YLabel'][val] for val in shuffle_idx]
    whole_data['EgoTrafficDist'] = [whole_data['EgoTrafficDist'][val] for val in shuffle_idx]

    # Split data into training and testing
    final_train_idx = int(np.floor(split_frac * data_len))
    train_idx_array = range(0,final_train_idx)
    test_idx_array  = range(final_train_idx, data_len)

    train_data = {}
    test_data  = {}

    train_data['Xtrain'] = whole_data['Xtrain'][train_idx_array,:,:,:]
    train_data['YLabel'] = [whole_data['YLabel'][val] for val in train_idx_array]
    train_data['EgoTrafficDist'] = [whole_data['EgoTrafficDist'][val] for val in train_idx_array]

    test_data['Xtrain'] = whole_data['Xtrain'][test_idx_array,:,:,:]
    test_data['YLabel'] = [whole_data['YLabel'][val] for val in test_idx_array]+
    test_data['EgoTrafficDist']  = [whole_data['EgoTrafficDist'][val] for val in test_idx_array]

    return (train_data, test_data)



BATCH_SIZE = 32
num_classes = 2

# Initialize dictionary var that will contain collection of
# training data
data_samples = {}
train_samples = {}
test_samples = {}

# Scan for all available training data
data_file_list = glob.glob('Training2018*')

for data_file in data_file_list:
    fid  = open(data_file,'rb')
    temp_data = pickle.load(fid)

    # Iterate over each of label on temp_data, if it is yellow or red set it to 0,
    # if it is green set it to 1
    temp_data['YLabel'] = [0 if (item == 0 or item == 1) else 1 for item in temp_data['YLabel']]

    fid.close()

    # Store it inside dictionary
    if data_samples.has_key('Xtrain'):
        data_samples['Xtrain'] = np.append(data_samples['Xtrain'], temp_data['Xtrain'], axis=0)
        data_samples['YLabel'].extend(temp_data['YLabel'])
        data_samples['EgoTrafficDist'].extend(temp_data['EgoTrafficDist'])
    else:
        data_samples = temp_data

# Split data into training and test/validation
train_samples, test_samples = split_data(data_samples, 0.9)
#train_samples = data_samples
#import copy
#test_samples = copy.copy(data_samples)

exploration_plot(train_samples)


# Shuffle and split data based on split_frac, where :
# training-data = split_frac *  whole_data
# test-data     = ( 1 - split_frac) * whole_data



def resize_image(image):
    return ktf.image.resize_images(image, (600, 800))

def bgr2rgb(img):
    img_res = np.copy(img)
    img_res[:,:,0] = img[:,:,2]
    img_res[:,:,2] = img[:,:,0]
    return img_res

def generator(samples, batch_size=BATCH_SIZE):
    # Samples is data-dictionary with keys as follow
    # Xtrain, YLabel, and EgoTrafficDist
    # Where Xtrain is a key to image set
    #       YLabel is a key to label for each image in image set (0 -> Red, 1 -> Yellow, 2 -> Green)
    #       EgoTrafficDist is a key to distance between ego to traffic light
    num_samples = samples['Xtrain'].shape[0]
    row_pix, col_pix, ch = samples['Xtrain'][0].shape

    # Convert YLabel value to np.array for convenience of indexing
    samples['YLabel'] = np.array(samples['YLabel'])
    samples['EgoTrafficDist'] = np.array(samples['EgoTrafficDist'])

    while 1:  # Loop forever so the generator never terminates

        shuffle_idx = random.sample(range(0,num_samples),num_samples)
        samples['Xtrain'] = samples['Xtrain'][shuffle_idx,:,:,:]
        samples['YLabel'] = samples['YLabel'][shuffle_idx]
        samples['EgoTrafficDist'] = samples['EgoTrafficDist'][shuffle_idx]

        for offset in range(0, num_samples, batch_size):
            Xtrain_samples = samples['Xtrain'][offset:offset + batch_size]
            YLabel_samples = samples['YLabel'][offset:offset + batch_size]

            img_processed = None
            light_states = YLabel_samples

            for orig_img in Xtrain_samples:
#                center_image = HistNormImage(img_orig)  # Perform histogram normalization
#                center_angle = float(batch_sample[3])

#                if (batch_sample[7]):  # If mirror flag is true
#                    for ch in range(center_image.shape[2]):
#                        center_image[:, :, ch] = np.fliplr(center_image[:, :, ch])
#                    center_angle = -center_angle



                if img_processed is None:
                    orig_img_size = orig_img.shape
                    img_processed = orig_img.reshape(1,row_pix, col_pix, ch)
                else:
                    img_processed = np.append(img_processed, orig_img.reshape(1,row_pix, col_pix, ch), axis=0)

            # trim image to only see section with road
#            X_train = np.array(images)
#            y_train = np.array(angles)
#            yield sklearn.utils.shuffle(X_train, y_train)
            # convert class vectors to binary class matrices
            y_train = keras.utils.to_categorical(YLabel_samples, num_classes)
            yield(img_processed, y_train)

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(test_samples, batch_size=len(test_samples['YLabel']))

row, col, ch = train_samples['Xtrain'].shape[1:]


# Start building up model
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Cropping2D(cropping=((0,0),(0,0)), input_shape=(row,col,ch)))
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row-0, col,ch),
        output_shape=(row-0, col, ch)))

# Resize image to fit NVIDIA net-work input  66x200x3
#model.add(Lambda(lambda image: ktf.image.resize_images(image, (row_new, col_new))))
model.add(Lambda(resize_image))

# First conv layer follow by relu
model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))

# Second conv layer follow by relu + drop-out
model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
model.add(Dropout(rate=0.3))
model.add(BatchNormalization())


# Third layer of conv layer follow by relu
model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
model.add(Dropout(rate=0.3))
model.add(BatchNormalization())

# Fourth layer of conv layer follow by relu
model.add(Conv2D(64,(3,3),strides=(1,1),activation='relu'))
model.add(Dropout(rate=0.2))
model.add(BatchNormalization())

# Fifth layer of conv layer follow by relu
model.add(Conv2D(64,(3,3),strides=(1,1),activation='relu'))
model.add(Dropout(rate=0.1))
model.add(BatchNormalization())

# Sixth layer Flattening the model, 1164
model.add(Flatten())

# Seventh layer fully connected layer (100 neurons output)
model.add(Dense(100, activation='relu'))

# Eight layer fully connected layer (50 neurons output)
model.add(Dense(50, activation='relu'))

# Ninth layer fully connected layer (10 neurons output)
model.add(Dense(10, activation='relu'))


# Final layer (1 neurons output)
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
#model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
#     validation_data=validation_generator,
#     nb_val_samples=85, nb_epoch=1, verbose=1)
model.fit_generator(train_generator,steps_per_epoch=len(train_samples['Xtrain'])/BATCH_SIZE, \
                    epochs=10,verbose=1,validation_data=validation_generator, \
                    validation_steps=1)

res = model.evaluate_generator(validation_generator,steps=1)

inp = next(validation_generator)
res2 = model.predict_on_batch(inp[0])

print(res2)

model.save('model1.h5')


# Testing: Run several plot randomly
sample_data = train_generator.next()
sample_no = random.sample(range(0,len(sample_data[0])),1)[0]

img = sample_data[0][sample_no]
label = sample_data[1][sample_no]
img_rgb = bgr2rgb(img)

plt.figure()
plt.imshow(img_rgb)
plt.axis('off')
plt.title(str(label))

# second-test
sample_data = train_generator.next()
sample_no = random.sample(range(0,len(sample_data[0])),1)[0]

img = sample_data[0][sample_no]
label = sample_data[1][sample_no]
img_rgb = bgr2rgb(img)

plt.figure()
plt.imshow(img_rgb)
plt.axis('off')
plt.title(str(label))




plt.show()

#train_generator2 = generator(train_samples, batch_size=BATCH_SIZE)


#validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)