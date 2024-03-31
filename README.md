[Read me.docx](https://github.com/MAmr9UH/Image-Classification--Convolutional-Neural-Network-/files/14817575/Read.me.docx)
Deep Learning Project: Happy/Sad Image Classifier
Setup and Installation
1-Install the required packages using pip:

!pip install tensorflow matplotlib opencv-python
2-Import TensorFlow and other necessary libraries:
import tensorflow as tf
import os
import cv2
from matplotlib import pyplot as plt
import imghdr
Remove Dodgy Images
1-Define the data directory and list the files within it:
data_dir = 'data'
os.listdir(data_dir)
2-Define the image extensions to filter out dodgy images:
image_exts = ['jpeg', 'jpg', 'bmb', 'png']
3-Use OpenCV and Matplotlib to process and visualize the images.

Load the Data
Load the image dataset using TensorFlow's image_dataset_from_directory function.





Preprocess Data
1-Scale the image data between 0 and 1:

data = data.map(lambda x, y: (x / 255, y))
2-Split the data into training, validation, and test sets.

Build and Train the Model
1-Create a convolutional neural network (CNN) model using TensorFlow's Keras API.
2-Train the model using the training and validation data.

Evaluate Performance
Evaluate the model's performance using metrics like precision, recall, and accuracy.
Save the Model
Save the trained model for future use.
![image](https://github.com/MAmr9UH/Image-Classification--Convolutional-Neural-Network-/assets/96629572/3b5587b0-2d0f-4ff4-adae-ba26eb0b0ab6)
