#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode :heavy_check_mark:

My project includes the following files:

* `model.py`: Contains the script to create and train the model.
* `drive.py`: File used for driving the car in autonomous mode.
* `model.h5`: Contains a trained convolution neural network.
* `writeup_report.md`: Summarizing my results and findings... and frustrations. :stuck_out_tongue_winking_eye:

####2. Submission includes functional code :heavy_check_mark:
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```
python drive.py model.h5
```

####3. Submission code is usable and readable :heavy_check_mark:

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code/functions works.

---
---
###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed :heavy_check_mark:

The model architecutre was inspired from the following sources:

* [NVIDIA CNN's Architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
* [Udacity's Self Driving Car Q & A | Behavior Cloning Youtube Video](https://www.youtube.com/watch?v=rpxZ87YFg0M).

Data normalization in the model was done using Kera's Lambda Layer and Kera's Convolutional Cropping Layer. This was done in the Q & A and it worked really well from the "get go" so I decided to continue with it throughout the project.

The model consists of depth 24, 36, 48, and 64 with 5x5 filters and 3x3 filters (`model.py` lines 193 - 206). This model includes the use of RELU layers within the Convolutional Layers and Fully Connected Layers (`model.py` lines 193 - 206). It also includes a deeper Fully Connected Layer with "neurons" descending from 1124 to 1 (`model.py` lines 209 - 218). Additionally, within the Fully Connected Layers, Dropout is introduced with a rate of `0.2` (`model.py` lines 209 - 218).


####2. Attempts to reduce overfitting in the model :heavy_check_mark:

To reduce overfitting in the model, Dropout was introduced in the dense layers. I originally started with a modest dropout rate (`0.5`) but, I kept having issues on sharp turns. Once I reduced it to the rate of `0.2`, it appeared to help.

The model was trained and validated on different data sets to ensure that the model avoided overfitting (`model.py` lines 32).


####3. Model parameter tuning

The model used an Adam Optimizer with a learning rate of `0.0001`.  (`model.py` line 219-220).


####4. Appropriate training data

The training data chosen for this project was actually the data provided from Udacity. I had tried training with my own recordings, which encompassed two full laps fowards and backwards with a combination of recovering from left and right sides of the road. Ultimately, the model I had created worked best the Udacity Data.

The data I had created as well was also very heavy, creation of the model took much longer becasue of the amount of data. I was unable to successfully use AWS and Floyd.

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...


###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

**Answer:**
The overall strategy for deriving a good model architecture was to find a balance between the architecture and the approriate data.

The first steps I implemented were assisted with the [Self Driving Car Q & A | Behavior Cloning Youtube Video](https://www.youtube.com/watch?v=rpxZ87YFg0M). We started with a simplistic network:

```
Convolution -> ReLU -> MaxPooling2D -> Convolution -> ReLU -> MaxPooling2D ->
Flatten -> Fully Connected -> Fully Connected -> Fully Connected
```
which did not work so well. Because of the bias for left turning in the simulator, it would ultimiately end up off the street.

Two things needed to improve, my dataset and probably my neural network. Based of the recommendation, I pursued the [NVIDIA Self Driving Car Model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) with some tweaks that I thought would help, learned from the previous project (Traffic Sign Classfier). To optimize the data, I practiced using the simiulator until I feel more comfortable recording my laps. I did 3 laps going the correct direction and then did 3 laps in the reverse to hopefully eliminate the left bias. Additionally, I did a few recordings for correcting the car when it was on the lane lines.

With all the data, I implemented a few other preprocessing tricks to help the model learn. I implemented a flip, shift, and a brightness adjustment before separating the into training and validation. It was around this stage that my model would beging to work **at times** and at other times would not work.

I then when back and started adding more training data for the edge cases that I seemed to be running into. For instance, recovering when going into the lane lines. Around this same time, I did some digging into what other options I had to modify the NVIDIA framework


####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

**Answer:**

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Anwser Done
**Answer:**
To capture good driving behavior, I first recorded 2 laps going in the correct direction on track one using center lane driving and two laps going in the opposite direction on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][/examples/left_track/center_2017_03_09_19_35_51_413.jpg]

I then recorded the video recovering from the right side of the road back to center so that the vehicle would learn to adjust itself. The following images show a glimpse of that:

![Recovery Lane Driving][/examples/edge_turns/center_2017_03_12_17_04_38_057.jpg]
![Recovery Lane Driving][/examples/edge_turns/center_2017_03_12_17_04_38_895.jpg]
![Recovery Lane Driving][/examples/edge_turns/center_2017_03_12_17_04_39_591.jpg]
![Recovery Lane Driving][/examples/edge_turns/center_2017_03_12_17_04_39_591.jpg]
![Recovery Lane Driving][/examples/edge_turns/center_2017_03_12_17_04_40_228.jpg]

After the entire collection process, I had x number of data points. I imported the data for the different data sources and ran `train_test_split` with a `20%` test size. From here I utilized the generator method that was presented to us in our lesson.  I then ran a routine for preprocessing which involed the following:

* Shifting the Image
* Flipping the Image
* Augmenting the Brightness of the Image

I used the training data for training the model and the validation set to help determine whether we were over or under fitting. I used the adam optimizer so that manually training the learning rate wasn't necessary.