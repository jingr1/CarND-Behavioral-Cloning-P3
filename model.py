import cv2
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Lambda

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
from os.path import join

from keras.callbacks import ModelCheckpoint, CSVLogger
from config import *
import csv
from preprocess import preprocess
# Read images and steering angles from training data
def read_csv_data(path = 'data/driving_log.csv'):
    '''
    Read all the data from csv file with padas,
    Not used in this project as it needs lots of memorys.
    '''
    df=pd.read_csv(path, sep=',',header=0)

    center_image_source_path = df['center']
    center_image_current_path = center_image_source_path.apply(lambda x: 'data/IMG/'+ x.split('/')[-1])
    center_images = center_image_current_path.apply(lambda x : cv2.imread(x))
    steering_angles = df['steering']

    X_train = np.array(center_images)
    y_train = np.array(steering_angles)

    return X_train, y_train


def split_train_val(csv_driving_data, test_size=0.2):
    """
    Splits the csv containing driving data into training and validation

    :param csv_driving_data: file path of Udacity csv driving data
    :return: train_split, validation_split
    """
    with open(csv_driving_data, 'r') as f:
        reader = csv.reader(f)
        driving_data = [row for row in reader][1:]

    train_data, val_data = train_test_split(driving_data, test_size=test_size, random_state=1)

    return train_data, val_data


def generate_data_batch(data, batchsize=CONFIG['batchsize'], data_dir='data', augment_data=True):
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
    while True: # Loop forever so the generator never terminates
        # shuffle data
        shuffled_data = shuffle(data)
        for offset in range(0,num_data,batchsize):
            batch_datas = data[offset:offset+batchsize]
            images = []
            angles = []
            for batch_data in batch_datas:
                center_image = preprocess(cv2.imread(join(data_dir,batch_data[0].strip())))
                center_angle = float(batch_data[3])
                images.append(center_image)
                angles.append(center_angle)
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
    model.add(Lambda(lambda x: x/255. - 0.5,input_shape=(NVIDIA_H, NVIDIA_W, CONFIG['input_channels'])))
    model.add(Conv2D(24, (5, 5), strides=(2, 2),activation = 'relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2),activation = 'relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2),activation = 'relu'))
    model.add(Conv2D(64, (3, 3),activation = 'relu'))
    model.add(Conv2D(64, (3, 3),activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    if summary:
        model.summary()

    return model

if __name__ == '__main__':
    # split udacity csv data into training and validation
    train_data, val_data = split_train_val(csv_driving_data='data/driving_log.csv')

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
    model.fit_generator(generator=generate_data_batch(train_data, augment_data=True),
                        steps_per_epoch=len(train_data)/CONFIG['batchsize'],
                        epochs=1,
                        validation_data=generate_data_batch(val_data, augment_data=False),
                        validation_steps=len(val_data)/CONFIG['batchsize'],
                        callbacks=[checkpointer, logger])
    model.save('model1.h5')
