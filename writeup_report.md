# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras(2.1.5) that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/model.png "Model Visualization"
[image2]: ./writeup/center_lane_driving.jpg "Grayscaling"
[image3]: ./writeup/recovery1.jpg "Recovery Image"
[image4]: ./writeup/recovery2.jpg "Recovery Image"
[image5]: ./writeup/recovery3.jpg "Recovery Image"
[image6]: ./writeup/Normal.jpg "Normal Image"
[image7]: ./writeup/fliped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 - A video recording of my vehicle driving autonomously one lap around the track.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model corresponding to the NVIDIA architecture described in paper "End to end learning for self-driving cars." which consists of 10 layers, including a normalization layer, a Cropping layer, 5 convolutional layers and 3 fully connected layers.

The first layer of the network performs image normalization. The normalizer is hard-coded and is not adjusted in the learning process. Performing normalization in the network allows the normalization scheme to be altered with the network architecture and to be accelerated via GPU processing.

The keras Cropping layer were used to crop each image to focus on only the portion of the image that is useful for predicting a steering angle.

The convolutional layers were designed to perform feature extraction and were chosen empirically
through a series of experiments that varied layer configurations. We use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers.

We follow the five convolutional layers with three fully connected layers leading to an output control value which is the steering angle. The fully connected layers are designed to function as a
controller for steering, but we note that by training the system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractor and which  serve as controller.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Fursermore, I used data from both track one and track two to make a more generalized model.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use Keras sequential model using tensorflow backend.

My first step was to use a convolution neural network model similar to the NVIDIA architecture described in paper "End to end learning for self-driving cars." I thought this model might be appropriate because it was trained to map raw pixels from a single front-facing camera directly to steering commands and learn to drive in traffic on local roads with or without lane markings and on
highways With minimum training data from humans the system.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error (0.000777809085402) on the training set but a high mean squared error(0.0532700421651) on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that it has a low mean squared error on both the training and validation sets.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially around the corner. To improve the driving behavior in these cases, I collect more data especially the turning data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a normalization layer, a Cropping layer, 5 convolutional layers following a dropout layer and 3 fully connected layers following a dropout layer.
Here is a visualization of the architecture .

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the road center when it tend to departure the lane .These images show what a recovery looks like starting from right side of the road back to center.

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would helpful combat the left turn bias. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had X number of data points. I then preprocessed this data by normalization and cropping.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by mean squared error tend to be stable. I used an adam optimizer so that manually training the learning rate wasn't necessary.
