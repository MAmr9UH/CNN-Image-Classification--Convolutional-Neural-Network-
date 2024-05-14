# Deep Learning Project: Happy/Sad Image Classifier

This project involves the creation of a convolutional neural network (CNN) using TensorFlow's Keras API to classify images as either 'Happy' or 'Sad'. Below are detailed instructions on setting up the project, preprocessing the data, building and training the model, and evaluating its performance.

## Setup and Installation

### Install Required Packages

Install the necessary Python packages using pip:

```bash
pip install tensorflow matplotlib opencv-python
Import Necessary Libraries
Import the required libraries for the project:
```
python

import tensorflow as tf
import os
import cv2
from matplotlib import pyplot as plt
import imgdhr

```

##Data Handling
#Remove Dodgy Images
Define the directory and filter out unwanted image formats:

```python

data_dir = 'data'
os.listdir(data_dir)
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
```

##use OpenCV and Matplotlib
Utilize OpenCV and Matplotlib for image processing and visualization.

##Preprocess Data
Scale Image Data
Scale the image data between 0 and 1 for neural network compatibility:
```
python

data = data.map(lambda x, y: (x / 255, y))
```


##Split Data
Split the data into training, validation, and test sets.

##Build and Train the Model
Create Model
Create a convolutional neural network (CNN) using TensorFlow's Keras API:
```
python

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    keras.layers.MaxPooling2D(2,2),
    ...
    keras.layers.Dense(2, activation='softmax')
])
```

##Train Model
Train the model using the training and validation data:
```
python

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_data, epochs=10, validation_data=validation_data)
```

##Evaluate Performance
Evaluate the model's performance using metrics like precision, recall, and accuracy.

##Save the Model
Save the trained model for future use:

python
```
model.save('happy_sad_model.h5')
```


##Conclusion

This deep learning project successfully demonstrated the capability of a convolutional neural network (CNN) to classify images into 'Happy' or 'Sad' categories using TensorFlow and Keras. The project utilized advanced machine learning techniques, including data preprocessing, model building, and real-time image classification with high accuracy. By integrating tools like OpenCV for image manipulation and Matplotlib for data visualization, the project effectively harnessed the power of Python for deep learning tasks. The classifier’s performance, evaluated on precision, recall, and accuracy, underscores the effectiveness of CNNs in automated image classification. This project not only enhances practical understanding of neural networks but also showcases the potential of deep learning in real-world applications, such as emotion recognition and human-computer interaction.
