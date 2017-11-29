#**Traffic Sign Recognition** 



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Training data distribution"
[image2]: ./examples/bar_after_generated.png "Training data distribution with augmented data"
[image3]: ./examples/keep_left_color.png "Original color"
[image4]: ./examples/keep_left_gray.png "Grayscaling"
[image5]: ./examples/top_k_prob.png "Top 5 Probabilty"
[image6]: ./examples/11.png "Clear image from wikipedia"
[image10]: ./img2/1.jpeg "Speed limit (30km/h)"
[image11]: ./img2/11.jpeg "Right-of-way at the next intersection"
[image12]: ./img2/12.jpeg "Priority road"
[image13]: ./img2/13.jpeg "Yeild"
[image14]: ./img2/14.jpeg "Stop"
[image15]: ./img2/18.jpeg "General caution"
[image16]: ./img2/25.jpeg "Road work"
[image17]: ./img2/27.jpeg "Pedestrians"
[image18]: ./img2/28.jpeg "Children crossing"
[image19]: ./img2/38.jpeg "Keep right"
[image20]: ./examples/featuremap1.png "Conv layer 1"
[image21]: ./examples/featuremap2.png "Relu layer 1"
[image22]: ./examples/featuremap3.png "Pooling layer 1"
[image23]: ./examples/featuremap4.png "Conv layer 2"
[image24]: ./examples/featuremap5.png "Relu layer 2"
[image25]: ./examples/featuremap6.png "Pooling layer 2"






## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. 
I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed. Some classes have around 2000 samples, while others only have around 200. So In the data pre-process section we'll first try to generate more data for those with a smaller size. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data.

1. convert to grayscale using: np.array(np.dot(rgb[...,:3], [0.299, 0.587, 0.114]))
2. normalize using: (pixel - 128)/ 128 



As a first step, I decided to convert the images to grayscale because it'll reduce number of features and increase training speed.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3] ![alt text][image4]

Then I normalized the image data because it'll make gradient descent running faster. 

I decided to generate additional data because the model has higher training accuracy and lower validation accuracy, so it is somewhat overfitting. 

The following techniques are applied to generate more data:

3. find the classes with less than 500 training sample, and do shifts by 2 pixel along the axis=(1,2), then add to original set. This will help make the trainig sample distribution even. 
4. do zoom by 1.2 then slice to correct image size
5. do zoom by 0.8 then do np.pad
6. do rotate by 5,-5,10,-10 degrees

Then we add all the generated data together, we get 328433 training samples, from original 34799. 

![alt text][image2]


####2. Describe what your final model architecture looks like 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution	 5x5   | 1x1 stride, valid padding, outputs 10x10x16    									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten      	| flatten both conv1 and conv2, combine outputs 1576  				|
| Fully connected		| input 1576, output 120        									|
| RELU					|												|
| Fully connected		| input 120, output 84        									|
| RELU					|												|
| Fully connected		| input 84, output 43        									|
| Softmax				|         									|

 


####3. Describe how you trained your model. 

Here are the details to train the model:

1. Optimizer: Adam
2. Batch size: 128
3. EPOCHs: 15
4. Initial learning rate: 0.001

I tried to do exponentially decay the training rate whenever validation accuracy decreases, by: 

```
	if (validation_accuracy < validation_accuracy_pre):
		rate = 0.5 * rate
``` 

Since I use a larger epoch, 15, I also tried to stop training when training accuracy is high enough by:

```
	if (1.0 - train_accuracy < 0.0001):
		print("Train accuracy 1.0, break")
		break
```

####4. Describe the approach taken for finding a solution 

My final model results were:

* training set accuracy of 1.0
* validation set accuracy of 0.955
* test set accuracy of 0.945

1. I first directly tried the LeNet, with no generated data, which gives about 0.9 accuracy.
2. Then I tried to add one more fully connected layer, which gives a higher training accuracy but even lower validation accuracy. I think add more layers makes the network overfitted, then just removed the extra layer and fall back to LeNet.
3. Since the architecture is already towards overfitting, I decided to focus on getting more training data in the following tests.
4. Step by step, I added rotation, zooming, shift, to get more data. Among which I found the zooming have most effect, then rotation, then shifting. 
5. Then I followed the paper to add the result of first convolutional layer together with the second one to the fully connected layer. This give a little bit more accuracy, which is the final result above. 

 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. 

Here are 10 German traffic signs that I download from google:

![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14] ![alt text][image15]
![alt text][image16] ![alt text][image17] ![alt text][image18]
![alt text][image19]

The "right of way" image is difficult to classify, because it have lots of rotation, and the image is too close to the right side. 

The "Generala caution" one have another sigh underneath it, so that'll cause confusion also. 

####2. Discuss the model's predictions on these new traffic signs 


Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop      		| Stop   									| 
| 30 km/h      		| 30km/h   									| 
| Right of way     			| Yield 										|
| Priority road					| Priority road											|
| Yield	      		| Yield					 				|
| Children crossing			| Children crossing      							|
| General caution			| No passing      							|
| Pedestrain			| General caution      							|
| Keep right			| Keep right      							|
| Road work			| Road work      							|


The model was able to correctly guess 7 of the 10 traffic signs, which gives an accuracy of 70%, compared with test set accuracy with 94.5%. Because this test sample is too small, only 10 images, so it's hard to say that the model will do so badly on other images. When I tried the more clear images from https://en.wikipedia.org/wiki/Road_signs_in_Germany, like the one with following, the model sometimes can have 100% accuracy. 

![alt text][image6]


####3. Describe how certain the model is when predicting 

The following image shows the top 5 probabilty of the testing images. For the correct predicted images, the model is pretty sure about the right label, with probability very close to 1.0, like the stop sign, 30km limit etc.

For the incorrect ones, The "right of way" image is a difficult one because it has lots of rotation, and the model is not that sure about it compared with other correctly labeled ones.

The "General Caution" image is incorrectly labeled with "No passing". I guess the german characters under the sign make the model confusing. 

The "General Caution" and "Pedestrian" kinds of pretty similar, so it's not that a surprise that it labeled "Pedestrain" with "General Caution".

![alt text][image5]



###  Visualizing the Neural Network 
####1. Discuss the visual output of your trained network's feature maps. 

I tried to visualize featuremap for the convolution layer 1 & 2, together with the Relu layer and max pooling layer, so total 6 images.

The convolutional layer maitains shape of the sign, and The relu layer makes the image to be like binary, either black or white. The max pooling layer dropped quite a lot information, make it quite difficult for humans to read the image. But I guess this is because the original image is pretty small, for real world images with high resolution, max pooling can help reduce lots of features.

Following are the featuremap images: 

![alt text][image20]
![alt text][image21]
![alt text][image22]
![alt text][image23]
![alt text][image24]
![alt text][image25]




