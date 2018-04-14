import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Cropping2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
from os.path import join

from keras.callbacks import ModelCheckpoint, CSVLogger
from config import *
import csv

def split_train_val(csv_driving_data, test_size=0.2):
    """
    Splits the csv containing driving data into training and validation

    :param csv_driving_data: file path of Udacity csv driving data
    :return: train_split, validation_split
    """
    with open(csv_driving_data, 'r') as f:
        reader = csv.reader(f)
        driving_data = [row for row in reader][1:]

    train_data, val_data = train_test_split(driving_data, test_size=test_size, random_state=0)

    return train_data, val_data


def generate_data_batch(data, batchsize=CONFIG['batchsize'], data_dir='track1/IMG', augment_data=True):
    '''
    Generator that indefinitely yield batches of training data from 'data' list.
    A batch of data is constituted by a batch of frames of the training track as well as the corresponding
    steering directions.

    :param data: list of training data in the format provided by Udacity
    :param batchsize: number of elements in the batch
    :param data_dir: directory in which frames are stored
    :param augment_data: if True, perform data augmentation on training data
    :return: X, Y which are the batch of input frames and steering angles respectively
    '''
    num_data = len(data)
    correction = CONFIG['correction']
    while True: # Loop forever so the generator never terminates
        # shuffle data
        shuffled_data = shuffle(data)
        for offset in range(0,num_data,batchsize):
            batch_datas = shuffled_data[offset:offset+batchsize]
            images = []
            angles = []
            for batch_data in batch_datas:
                center_image = cv2.imread(join(data_dir,batch_data[0].split('\\')[-1]))
                left_image = cv2.imread(join(data_dir,batch_data[1].split('\\')[-1]))
                right_image = cv2.imread(join(data_dir,batch_data[2].split('\\')[-1]))
                steering_center = float(batch_data[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                use_three_camera = False
                if use_three_camera:
                    raw_images = [center_image,left_image,right_image]
                    raw_angles = [steering_center,steering_left,steering_right]
                else:
                    raw_images = [center_image]
                    raw_angles = [steering_center]
                images.extend(raw_images)
                angles.extend(raw_angles)
                if augment_data:
                    #Augument the data with Flipping Images
                    images_flipped,angles_flipped = [],[]
                    for image,angle in zip(raw_images,raw_angles):
                        images_flipped.append(np.fliplr(image))
                        angles_flipped.append(-angle)
                    images.extend(images_flipped)
                    angles.extend(angles_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
def get_model(summary = True):
    '''
    Build a convolution neural network in Keras, this model corresponding to the
    NVIDIA architecture described in paper "End to end learning for self-driving cars."

    :param summary: show model summary
    :return: keras Model of NVIDIA architecture
    '''
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
    model.add(Cropping2D(cropping = ((70,25),(0,0))))
    model.add(Conv2D(24, (5, 5), strides=(2, 2),activation = 'relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2),activation = 'relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2),activation = 'relu'))
    model.add(Conv2D(64, (3, 3),activation = 'relu'))
    model.add(Conv2D(64, (3, 3),activation = 'relu'))
    model.add(Dropout(0.75))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    if summary:
        model.summary()

    return model

if __name__ == '__main__':
    # split udacity csv data into training and validation
    train_data, val_data = split_train_val(csv_driving_data='track1/driving_log.csv')

    # get network model and compile it (default Adam opt)
    model = get_model(summary=True)
    model.compile(optimizer='adam', loss='mse')

    # json dump of model architecture

    if not os.path.isdir('logs'):
        os.makedirs('logs')
    with open('logs/model.json', 'w') as f:
        f.write(model.to_json())

    # define callbacks to save history and weights
    if not os.path.isdir('checkpoints'):
        os.makedirs('checkpoints')
    checkpointer = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.3f}.hdf5')
    logger = CSVLogger(filename='logs/history.csv')

    # start the training
    model.fit_generator(generator=generate_data_batch(train_data, augment_data=False),
                        steps_per_epoch=len(train_data)/256,
                        epochs=30,
                        validation_data=generate_data_batch(val_data, augment_data=False),
                        validation_steps=len(val_data)/256,
                        callbacks=[checkpointer, logger])
    model.save('model.h5')
