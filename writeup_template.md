# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Figures/Sample.png "Sample Image"
[image2]: ./examples/Figures/BarGraph.png "Bar Graph of Frequencies"
[image3]: ./examples/Figures/Preprocess.png "Preprocessing Steps"
[image4]: ./examples/Figures/new1.png "Traffic Sign 1"
[image5]: ./examples/Figures/new2.png "Traffic Sign 2"
[image6]: ./examples/Figures/new3.png "Traffic Sign 3"
[image7]: ./examples/Figures/new4.png "Traffic Sign 4"
[image8]: ./examples/Figures/new5.png "Traffic Sign 5"
[image9]: ./examples/Figures/new6.png "Traffic Sign 6"
[image10]: ./examples/Figures/new7.png "Traffic Sign 7"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/esouliot/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy ndarray.shape method, and Python's built-in set() method to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is a sample image from the training set, representative of the types of images to be fed into the LeNet implementation

![Sample Image][image1]

This bar graph shows the frequency of a given image tag type, some tags appear far more frequently in the set than others
![Bar Graph of Frequencies][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

1.) First, I converted the images to grayscale [Luma](https://en.wikipedia.org/wiki/Luma_(video)) using matrix multiplication. 

2.) Then, I used the Histogram Equalization method from the [Sci-Kit Image Exposure module](http://scikit-image.org/docs/dev/api/skimage.exposure.html) to normalize the pixel values

3.) Finally, I reshaped the modified images to fit the 4D shape input for the LeNet function


![PreProcessing Steps][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I was able to find success with most of the original steps from the LeNet architecture. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Activation					|	RELU											|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16  |
| Activation		| RELU        									|
| Max pooling        | 2x2 stride, outputs 5x5x16        									|
|	Flattening					|	Outputs 400x1 from 5x5x16											|
|	Activation					|	RELU											|
| Fully connected | Input 400, Output 120 |
| Activation | RELU |
| Fully connected | Input 120, Output 84 |
| Activation | RELU |
| Fully connected | Input 84, Output 43 |
| Return logits | |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the provided code from the [LeNet lab](https://github.com/esouliot/CarND-LeNet-Lab/blob/master/LeNet-Lab.ipynb)

* Loss function: [Softmax cross entropy with logits](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits)
* Loss Optimizer: [Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) was used
* Batch Size: 64
* Number of training epochs: 20
* Learning rate: 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 99.5%
* validation set accuracy of 93.2% 
* test set accuracy of 91.1%

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?

The LeNet architecture was used throughout, with the only change to the layers of the network being in the padding type of the max pooling layers (2x2 valid padding to 2x2 same padding)

* What were some problems with the initial architecture?

Not problems with the architecture, per se, but problems to do with tuning and data preprocessing (i.e., hyperparameter tuning, image grayscaling and normalizing)

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

The unchanged LeNet architecture with 10 training epochs on the raw images gave a validation accuracy around 89%, as noted in the pre-project documentation. While that level of accuracy is not to the criterion of the project, it still means that it can correctly predict approximately 89 of 100 images fed to it in the validation step. To meet the passing validation accuracy of >= 93%, I took to preprocessing the images, as mentioned in previous sections, and to changing the epochs, learning rate, and batch size hyperparameters until the model gave a high enough validation accuracy.

* Which parameters were tuned? How were they adjusted and why?

Epochs: Changed from 10 to 20

A simple explanation as to why I trained longer - more training epochs gives the loss optimizer more chances to "gradient-descend" and change the weights to give the lowest cross-entropy

Batch size: Changed from 128 to 64

This parameter was tuned by trial and error, and a smaller batch gave better results than larger batches. But, the improvement may be related to the number of epochs used, since a smaller batch size means that more batches were run, and with more batches come more iterations of the loss optimizer.

Learning rate: Stayed the same at 0.001

I tested a learning rate of 0.01, which gave some training epochs of >=93% validation accuracy, but it also tended to "over-step" and go in the wrong gradient direction, oftentimes resulting in decreasing validation accuracy for later epochs. 

In hindsight, perhaps even 0.001 is too high for later epochs, because as can be seen in my project Jupyter notebook, the validation accuracy of my model fluctuates around 0.93 in the final 5-or-so epochs, sometimes dropping from higher values in the step from the 19th to the 20th epoch.This issue could  possibly be remediated using a learning-rate decay, or a simple iterative learning rate decrease.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

* Use of convolutions

As in the numeral classification problem of the [MNIST](https://www.tensorflow.org/get_started/mnist/pros) dataset, convolutions are of use in classifying traffic signs, because for a given grayscaled image, a neural network can work its way up from pixel gradients to lines, to shapes, to combinations of shapes, and then giving tags to said combinations.

* Use of max pooling

Because I made the choice to grayscale the images, I believe that max pooling would work better for shape detection in this task, since the features of a given sign could be defined as combinations of edges, demarcated by changes in pixel gradient, comparable to the approach used in the [Lane Lines](https://github.com/esouliot/CarND-LaneLines-P1/blob/master/P1.ipynb) project with Canny edge detection. 

* Use of dropout (or lack thereof)

While the model slightly overfitted the training data (final training accuracy of 0.995, final validation accuracy of 0.932), it was not overfit enough to justify the use of a dropout in the fully connected layers. 


If a well known architecture was chosen:

* What architecture was chosen?

A slightly modified version of LeNet was used

* Why did you believe it would be relevant to the traffic sign application?
The imagery of traffic signs are not unlike drawn numerals (in fact, speed limit signs include numerals), and since LeNet gave a 99% test accuracy on the MNIST dataset of drawn numerals, it isn't unreasonable to think LeNet would work well in classifying traffic signs. 

That, and Yann LeCun used convolutional networks to [classify traffic signs with 99.17% accuracy](yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 A 91.1% test accuracy on completely untrained images is fairly accurate, with minimal changes to the network architecture.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image4] 

Nothing difficult about this sign. Taken from a high-resolution image, photographed on a clear, sunny day.

![alt text][image5]

Because of the grassy background to this image, the grayscaled normalized version exhibits some nose around the sign

![alt text][image6]

Looking back in this image, it seems that it may not actually be a German speed limit sign, because the numeric font used is not 

![alt text][image7] 

![alt text][image8] 

![alt text][image9] 

![alt text][image10]



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


