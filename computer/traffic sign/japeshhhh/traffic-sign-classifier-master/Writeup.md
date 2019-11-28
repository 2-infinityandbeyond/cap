# **Traffic Sign Recognition** 

## Writeup

### Author: Sergey Morozov

---

**Build a Traffic Sign Recognition Project**

The goals/steps of this project are the following:
* Load the dataset (included into this repository in [traffic-sign-data](./traffic-sign-data) directory)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./traffic-sign-images/11-Right-of-way-at-the-next-intersection.jpg "Right of way at the next intersection"
[image2]: ./traffic-sign-images/12-Priority-road.jpg "Priority road"
[image3]: ./traffic-sign-images/14-Stop.jpg "Stop"
[image4]: ./traffic-sign-images/15-No-vehicles.jpg "No vehicles"
[image5]: ./traffic-sign-images/1-Speed-limit-30-km-h.jpg "Speed limit 30 km/h"
[image6]: ./writeup-images/before.png
[image7]: ./writeup-images/normalization.png
[image8]: ./writeup-images/grayscale.png

## Rubric Points
Here I consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) 
individually and describe how I addressed each point in my implementation.

The project code can be found in [Traffic_Sign_Classifier.ipynb](./Traffic_Sign_Classifier.ipynb).
The execution results can be found in [Traffic_Sign_Classifier.html](./Traffic_Sign_Classifier.html).

### Dataset Exploration

#### Dataset Summary

The basic statistics such as images shapes, number of traffic sign categories, 
number of samples in training, validation and test image sets are presented in the 
*Step 1: Dataset Summary & Exploration* section, *A Basic Summary of the Dataset* subsection. 

#### Exploratory Visualization

In section *Step 1: Dataset Summary & Exploration*, *An Exploratory Visualization of the Dataset* subsection
you can find example images from each category.

There is also a colorful histogram representing a count of samples per each category in the training set. 
As we can see, some categories have
many samples (maximum: 2010) and some categories have a fewer number of samples (minimum: 180).
I describe how to  "equalize" number of samples in each category in the later sections.

### Design and Test a Model Architecture

#### Preprocessing

Image preprocessing can be found in *Step 2: Design and Test a Model Architecture* section, 
*Pre-process the Data Set (normalization, grayscale, and so forth)* subsection.

There are some transformations required to be performed on each image to feed it to the neural network.
1. Normalize RGB image. It is done to make each image "look similar" to each other, to make input consistent.
2. Convert RGB image to grayscale. It was observed that neural network performs slightly better on the grayscale images.
It also may be wrong observations.

I was also tried to use adaptive histogram equalization for improving the local contrast and 
enhancing the definitions of edges in each region of an image but it decreased the performance of the
network, so only normalization and grayscale conversion were used in the final implementation.

Below are the original image, image after normalization and grayscale image.

![alt text][image6] ![alt text][image7] ![alt text][image8]

The training set was expanded with more data. The intent was to make a count of samples in each category equal,
categories containing a smaller number of samples were expanded with more, duplicate, images. 
The probabilities to get images from each category during training became equal. It dramatically improved
neural network performance. The size of training set became 43 * 2010 = 86430 samples.

#### Model Architecture

The model architecture is defined in *Step 2: Design and Test a Model Architecture*, *Model Architecture* subsection.
The architecture has 5 layers - 2 convolutional and 3 fully connected.
It is LeNet-5 architecture with only one modification - dropouts were added between the layer #2 and layer #3,
the last convolutional layer and the first fully connected layer. It was done to prevent neural network from overfitting
and significantly improved its performance as a result.

Below is the description of model architecture.

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 gray scale image                      | 
| Convolution 5x5       | 1x1 stride, same padding, outputs 28x28x6     |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 14x14x6                   |
| Convolution 5x5       | 1x1 stride, same padding, outputs 10x10x16    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| Flatten               | output 400                                    |
| Drop out              |                                               |
| Fully connected       | output 120                                    |
| RELU                  |                                               |
| Fully connected       | output 84                                     |
| RELU                  |                                               |
| Fully connected       | output 43                                     |


#### Model Training

The model is using Adam optimizer to minimize loss function. It worked better than stochastic gradient descent. 
The following hyperparameters were defined and **carefully** adjusted:
```
# learning rate; with 0.001, 0.0009 and 0.0007 the performance is worse 
RATE       = 0.0008

# number of training epochs; here the model stops improving; we do not want it to overfit
EPOCHS     = 30

# size of the batch of images per one train operation; surprisingly with larger batch sizes neural network reached lower performance
BATCH_SIZE = 128

# the probability to drop out the specific weight during training (between layer #2 and layer #3)
KEEP_PROB  = 0.7

# standart deviation for tf.truncated_normal for weights initialization
STDDEV     = 0.01
```

#### Solution Approach

In average the trained model correctly classifies traffic on the validation set in 96% cases, 
on the training set in 99% cases and on the test set in 93.5% cases. The decimal part mostly depends on the
data shuffling that is random. The best result I observed was 97% of correct classifications on the validation set; 
unfortunately, that model was overfitted and performed worse on other images.

The code can be found in section *Step 2: Design and Test a Model Architecture*, 
subsection *Train, Validate and Test the Model*.


In general, I believe LeNet-5 architecture fits good for this task since there are 
a lot of kinds of traffic signs that contain letters and symbols (LeNet-5 is good for symbols classification). 
Also, to improve an accuracy of classifications of traffic signs
with speed numbers, like 30 km/h, 70 km/h, additional convolutional layers could be added. 
(this is only a hypothesis)
 

### Test a Model on New Images

#### Acquiring New Images

Here are five German traffic signs that were found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] ![alt text][image4] ![alt text][image5]

The first image, "right of way at the next intersection", might be difficult to classify because it has a picture 
in the middle that is not very visible at the resolution 32x32. 
The last image may be difficult to classify because it contains text in the middle with a specific speed limit. 
There are a lot of categories in the "speed limit" super-category. 
At the 32x32 resolution and low brightness or contrast, they are hardly distinguishable.

#### Performance on New Images

Surprisingly, on these five images, the performance of the predictions was 100%. However, when the similar model was
trained with 50 epochs, there was a mistake with "speed limit 30 km/h" traffic sign, the model was overfitted.
There may be mistakes on other types of images. With other models, I had a problem with "end of all speed and passing limits"
traffic sign classification. Also, the results on the test set were not perfect (93.5%), so, certainly, there are images
somewhere on the web that this model will not be able to recognize.

The code can be found in the section *Step 3: Test a Model on New Images*, *Load and Output the Images* subsection.

Here are the results of the prediction:

|                  PREDICTED                  |                   ACTUAL                    |
|:-------------------------------------------:|:-------------------------------------------:|
| 1            Speed limit (30km/h)           | 1            Speed limit (30km/h)           |
| 12              Priority road               | 12              Priority road               |
| 11  Right-of-way at the next intersection   | 11  Right-of-way at the next intersection   |
| 14                   Stop                   | 14                   Stop                   |
| 15               No vehicles                | 15               No vehicles                |

#### Model Certainty - Softmax Probabilities

The code for making predictions on my final model is located in the *Step 3: Test a Model on New Images*, 
*Top 5 Softmax Probabilities For Each Image Found on the Web* subsection.

The model was quite certain about the four images. It was also pretty certain about "no vehicles" traffic signal,
but not totally. Below are top 5 softmax probabilities for "**no vehicles**" traffic sign.

| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 0.756710649           | No vehicles                                   | 
| 0.122047313           | Speed limit (30km/h)                          |
| 0.0512931943          | Priority road                                 |
| 0.0313977301          | Stop                                          |
| 0.0146821784          | No passing                                    |
