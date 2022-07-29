#!/usr/bin/env python
# coding: utf-8

# #Image classification

# **Prerequisite**
# 
# Upload the test image. Given in the folder (cat)

# **Instruction**
# 
# Image classification is an important task for computer vision application. Image classification
# algorithms made advancement from traditional feature-based methods to deep learning-based
# techniques. Deep learning, particularly the convolutional neural network, has been a success story in
# the last decade and significantly improved classification accuracy. In this task, you need to build a CNN
# architecture and optimise it for classification. Following tasks are to be carried out. You are
# encouraged to use Google Colab (https://colab.research.google.com/) with the GPU option enabled
# where suitable. Please use Keras deep learning framework for this part of the assignment. 

# 1. Load the CIFAR10 small images classification dataset from Keras inbuilt datasets(https://keras.io/api/datasets/cifar10/). Display 10 random images from each of the 10
# classes (the images should change in every run).
# 
# 
# 

# In[1]:


from keras.datasets import cifar10


# In[2]:


import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from IPython.display import Image
from keras.layers import Dropout


# In[3]:


def load_dataset():
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	return trainX, trainY, testX, testY


# In[4]:


trainX, trainY, testX, testY = load_dataset()


# In[5]:


cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# In[6]:


import random

randomlist = []
uniqueIndex = []
MAX_LIMIT = 5000

index = 0
while index < len(cifar10_classes):
  num = random.randint(0,MAX_LIMIT)
  label = [trainY[num][0]]
  if label not in randomlist:
    randomlist.append(label[0])
    uniqueIndex.append(num)
    index = index+1

#print(randomlist)
#print(uniqueIndex)

# plot the images WRT uniqueIndex
pyplot.figure(figsize=(10, 10))
for i in range(10):
  pyplot.subplot(5, 5, i + 1)
  pyplot.imshow(trainX[uniqueIndex[i]])
  pyplot.title(cifar10_classes[trainY[uniqueIndex[i]][0]])
  pyplot.axis("off")
# show the figure
pyplot.show()


# 2. For the classification (10 image classes), write Python code to create a basic CNN network of 
# your choice (can be anything from practical 7, LeNet, AlexNet etc.)

# In[7]:


# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model


# In[8]:


# scale pixels
def prep_pixels(train, test):

	train_norm = train.astype('float32')
	test_norm = test.astype('float32')

	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0

	return train_norm, test_norm


# In[9]:


# plot diagnostic learning curves
def summary(history, epoch):

  pyplot.subplot(211)
  pyplot.title('Cross Entropy Loss')
  pyplot.plot(history.history['loss'], color='blue', label='train')
  pyplot.plot(history.history['val_loss'], color='orange', label='test')

  pyplot.subplot(212)
  pyplot.title('Classification Accuracy')
  pyplot.plot(history.history['accuracy'], color='blue', label='train')
  pyplot.plot(history.history['val_accuracy'], color='orange', label='test')

  filename = "epoch_"+epoch+ '_plot.png'
  pyplot.savefig(filename)
  pyplot.close()


# In[10]:


# evaluating a model
def trainModel(epochsValue):

  trainX, trainY, testX, testY = load_dataset()
  trainY = to_categorical(trainY)
  testY = to_categorical(testY)

  trainX, testX = prep_pixels(trainX, testX)

  model = define_model()

  history = model.fit(trainX, trainY, epochs=epochsValue, batch_size=64, validation_data=(testX, testY), verbose=0)

  _, acc = model.evaluate(testX, testY, verbose=0)
  print('> %.3f' % (acc * 100.0))
  test_loss, test_acc = model.evaluate(testX,  testY, verbose=2)

  summary(history,str(epochsValue))
  


# 3. Train and test the network and report the training loss, training accuracy and test accuracy for 
# various epochs.

# In[11]:


trainModel(20)


# In[27]:


Image('/content/epoch_20_plot.png')


# In[26]:


trainModel(25)


# In[28]:


Image('/content/epoch_25_plot.png')


# 4.Improve the architecture by changing the parameters, including but not limited to, learning 
# rate, epochs, size of the convolution filters, use of average pooling or max-pooling etc. 

# Added more MaxPool and Dropout

# In[13]:


def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model


# In[14]:


trainModel(80)
# try with 100


# In[15]:


Image('/content/epoch_80_plot.png')


# 5.Improve the architecture by introducing more convolutional and corresponding subsampling 
# layers

# In[16]:


def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.3))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model


# In[17]:


trainModel(100)


# In[18]:


Image('/content/epoch_100_plot.png')


# 6. Your final code should accept single image on the trained network and produce the output 
# class. 

# In[19]:


define_model().summary()


# In[31]:


# save model
model = define_model()
model.save('final_model.h5')


# In[32]:


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


# In[33]:


# load and prepare the image
def load_image(filename):
	img = load_img(filename, target_size=(32, 32))
	img = img_to_array(img)
	img = img.reshape(1, 32, 32, 3)
	img = img.astype('float32')
	img = img / 255.0
	return img


# In[34]:


#Try with unseen image
def identifyImage(fileName):
  img = load_image(fileName)
  model = load_model('final_model.h5')
  result = model.predict_classes( )
  Image(fileName)
  print(cifar10_classes[result[0]])


# In[42]:


Image('/content/cat.jpg')


# In[43]:


identifyImage('/content/cat.jpg')

