from config import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
def preprocess(frame_bgr, verbose=False):
    """
    Perform preprocessing steps on a single bgr frame.
    These inlcude: cropping, resizing, eventually converting to grayscale

    :param frame_bgr: input color frame in BGR format
    :param verbose: if true, open debugging visualization
    :return:
    """
    # set training images resized shape
    h, w = CONFIG['input_height'], CONFIG['input_width']

    # crop image (remove useless information)
    frame_cropped = frame_bgr[CONFIG['crop_height'], :, :]

    # resize image
    frame_resized = cv2.resize(frame_cropped, dsize=(w, h))

    # eventually change color space
    if CONFIG['input_channels'] == 1:
        frame_resized = np.expand_dims(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)[:, :, 0], 2)

    if verbose:
        plt.figure(1), plt.imshow(cv2.cvtColor(frame_bgr, code=cv2.COLOR_BGR2RGB))
        plt.figure(2), plt.imshow(cv2.cvtColor(frame_cropped, code=cv2.COLOR_BGR2RGB))
        plt.figure(3), plt.imshow(cv2.cvtColor(frame_resized, code=cv2.COLOR_BGR2RGB))
        plt.show()

    return frame_resized.astype('float32')
