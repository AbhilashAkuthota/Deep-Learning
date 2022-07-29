#!/usr/bin/env python
# coding: utf-8

# # Image denoising

# **Prerequisite**
# 
# Upload pretrainned model (autoencoder_model.h5) and Berkeley Segmentation Dataset, given in the folder

# Image denoising is a fundamental image processing problem and the basis for a pre-processing step for many advanced computer vision tasks.

# In[1]:


import matplotlib.pyplot as plt
from skimage.util import random_noise
import numpy as np
import cv2
from PIL import Image, ImageFilter
get_ipython().run_line_magic('matplotlib', 'inline')


#  1. To write codes with the following denoising methods (You can make use of any library you want).
# * Mean filter
# * Median filter 
# * Wavelet 
# * Deep learning (you are free to choose any pre-trained model you want – but you need to justify why did you select this model). You are not expected to train a new model for this part. 

# **Mean filter**
# 
# The mean filter is used to blur an image in order to remove noise. The idea of mean filtering is simply to replace each pixel value in an image with the mean (`average') value of its neighbors.

# In[2]:


def applyMeanFiler(fileName):
  image = cv2.imread(fileName)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
  noisy = random_noise(image, var=0.155**2)  

  new_image = cv2.blur((noisy*255).astype(np.uint8),(9, 9))
  return image, noisy,new_image


# **Median filter**
# 
# The median filter is a non-linear digital filtering technique, often used to remove noise from an image or signal. 
# The main idea of the median filter is to run through the signal entry by entry, replacing each entry with the median of neighboring entries. The pattern of neighbors is called the "window", which slides, entry by entry, over the entire signal.
# 
# The ‘medianBlur’ function from the Open-CV library can be used to implement a median filter.

# In[3]:


def applyMedianFilter(fileName):
  image = cv2.imread(fileName) 
  noisy = random_noise(image, var=0.155**2)
  new_image = cv2.medianBlur((noisy*255).astype(np.uint8), 9)
  return image, noisy, new_image


# **Wavelet**
# 
# Wavelet denoising relies on the wavelet representation of the image. The noise is represented by small values in the wavelet domain which are set to 0.
# 

# In[4]:


from skimage.restoration import denoise_wavelet


# In[5]:


def applyWaveletFilter(fileName):
  image = cv2.imread(fileName)  
  noisy = random_noise(image, var=0.155**2)
  new_image = denoise_wavelet(noisy, multichannel=True, rescale_sigma=True)
  new_image = np.array(new_image*255, dtype=np.uint8)
  return image, noisy, new_image


# **Deep learning**
# 
# Deep learning (you are free to choose any pre-trained model you want – but you need
# to justify why did you select this model). You are not expected to train a new model
# for this part.

# In[6]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model


# In[36]:


def preprocess(array):
    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array

def noise(array):
    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )
    return np.clip(noisy_array, 0.0, 1.0)

def display(array1, array2):
    n = 1
    #indices = np.random.randint(len(array1), size=n)
    #print (indices)
    images1 = array1[[4902], :]
    images2 = array2[[4902], :]

    plt.figure(figsize=(20, 20))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        plt.subplot(5, 5, 1)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        plt.title('Original')
        plt.axis("off")

        plt.subplot(5, 5, 2 )
        plt.imshow(image1.reshape(28, 28))
        plt.title('Noisy')
        plt.gray()
        plt.axis("off")

        plt.subplot(5, 5, 3 )
        plt.imshow(image2.reshape(28, 28))
        plt.title("Autoencoder Denoisy")
        plt.axis("off")
        plt.gray()
        plt.show()

    plt.show()

def getDLImages(array1, array2):
    n = 1
    #indices = np.random.randint(len(array1), size=n)
    #print (indices)
    images1 = array1[[4902], :]
    images2 = array2[[4902], :]

    for i, (image1, image2) in enumerate(zip(images1, images2)):
      return image2.reshape(28, 28), image1.reshape(28, 28), image2.reshape(28, 28)
  


# In[8]:


(train_data, _), (test_data, _) = mnist.load_data()
train_data = preprocess(train_data)
test_data = preprocess(test_data)
noisy_train_data = noise(train_data)
noisy_test_data = noise(test_data)


# In[2]:


# loading whole model
from keras.models import load_model
autoencoder = load_model('/autoencoder_model.h5')


# 2. The input to your code will be original and noisy images. The output will be denoised images

# `We will pass the original image to the filter. And filter method will be created the noisy image for the same passed image. And then noisy image will be input for the denoisy image.`

# In[11]:


# This method will take original image and filer name for applying filter and show the images end of the execution
def applyFilerOnImage(fileName, filter):
  if (filter == 'Mean'):
    orignal, noisy, denoisy = applyMeanFiler(fileName)
  elif (filter == 'Median'):
    orignal, noisy, denoisy = applyMedianFilter(fileName)
  elif (filter == 'Wavelet'):
    orignal, noisy, denoisy = applyWaveletFilter(fileName)
   
  plt.figure(figsize=(20, 20))
  plt.subplot(5, 5, 1)
  plt.imshow(cv2.cvtColor(orignal, cv2.COLOR_BGR2GRAY))
  plt.title('Original')
  plt.axis("off")

  plt.subplot(5, 5, 2 )
  plt.imshow(cv2.cvtColor((noisy*255).astype(np.uint8), cv2.COLOR_BGR2GRAY))
  plt.title('Noisy')
  plt.axis("off")

  plt.subplot(5, 5, 3 )
  plt.imshow(cv2.cvtColor(denoisy, cv2.COLOR_BGR2GRAY))
  plt.title(filter+" Denoisy")
  plt.axis("off")
  plt.show()


# In[12]:


#Applying Mean Filer
applyFilerOnImage('/content/image_1.jpg','Mean')


# In[13]:


#Applying Median Filer
applyFilerOnImage('/content/image_1.jpg','Median')


# In[14]:


#Applying Wavelet Filer
applyFilerOnImage('/content/image_1.jpg','Wavelet')


# In[38]:


# This used by Deep learning for listing the original and denoisy image.
# In the result 
predictions = autoencoder.predict(noisy_test_data)
display(noisy_test_data, predictions)


# 3. Compare the original and denoised images using the following metrics (you are free to use 
# any library):
# * Mean Squared Error (MSE)
# * Structural SIMilarity (SSIM) index

# In[40]:


# For SSIM using skimage
from skimage.metrics import structural_similarity as ssim


# In[41]:


# For MSE
def mse(imageA, imageB):
  err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
  err /= float(imageA.shape[0] * imageA.shape[1])
  return err


# In[62]:


def calMSEnSSIM(imageA, imageB):
  imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
  imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
  m = mse(imageA, imageB)
  s = ssim(imageA, imageB)
  return m , s


# In[63]:


# This method will use the MSE and SSIM algo for comparing the original image with denoisy image 
# and it will publish the result
def compareImages(imageA, imageB, title):
  m , s = calMSEnSSIM(imageA, imageB)
  imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
  imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
  fig = plt.figure(title)
  plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
  ax = fig.add_subplot(1, 2, 1)
  plt.imshow(imageA, cmap = plt.cm.gray)
  plt.axis("off")
  ax = fig.add_subplot(1, 2, 2)
  plt.imshow(imageB, cmap = plt.cm.gray)
  plt.axis("off")
  plt.show()


# In[64]:


# This method will use the MSE and SSIM algo for comparing the original image with denoisy image 
# and it will publish the result
def compareOrigNoisyImage(fileName,filter):
  if (filter == 'Mean'):
    orignal, noisy, denoisy = applyMeanFiler(fileName)
  elif (filter == 'Median'):
    orignal, noisy, denoisy = applyMedianFilter(fileName)
  elif (filter == 'Wavelet'):
    orignal, noisy, denoisy = applyWaveletFilter(fileName)

  compareImages(orignal, denoisy, filter)


# In[65]:


#Compare the original and denoised images WRT MSE and SSIM
compareOrigNoisyImage('/content/image_1.jpg','Mean')


# In[66]:


#Compare the original and denoised images WRT MSE and SSIM
compareOrigNoisyImage('/content/image_1.jpg','Median')


# In[67]:


#Compare the original and denoised images WRT MSE and SSIM
compareOrigNoisyImage('/content/image_1.jpg','Wavelet')


# In[69]:


#Compare the original and denoised images WRT MSE and SSIM
original, noisy, denoisy = getDLImages(noisy_test_data, predictions)
m = mse(original, denoisy)
s = ssim(original, denoisy)
fig = plt.figure('Autoencoder')
plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
ax = fig.add_subplot(1, 2, 1)
plt.imshow(imageA, cmap = plt.cm.gray)
plt.axis("off")
ax = fig.add_subplot(1, 2, 2)
plt.imshow(imageB, cmap = plt.cm.gray)
plt.axis("off")
plt.show()


# 4. Generate and report results (some sample images and graphs/tables) using the given dataset 
# of 25 original and noisy images.
# 
# `Report has beend generated WRT to (MSE and SSIM) for the above used filter (Mean, Median and Wavelet). Used the same images which are defined in the The Berkeley Segmentation Dataset`

# In[ ]:


# For concise the report for the given dataset image
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from PIL import Image
def conciseReportFor(fileName):
  orignal, noisy, denoisy = applyMeanFiler(fileName)
  mean_mse, mean_ssim = calMSEnSSIM(orignal, denoisy)
  orignal, noisy, denoisy = applyMedianFilter(fileName)
  median_mse, median_ssim = calMSEnSSIM(orignal, denoisy)
  orignal, noisy, denoisy = applyWaveletFilter(fileName)
  wavelet_mse, wavelet_ssim = calMSEnSSIM(orignal, denoisy)
  l = [["Mean", mean_mse, mean_ssim], ["Median", median_mse, median_ssim], ["Wavelet", wavelet_mse, wavelet_ssim]]
  table = PrettyTable(['Filter', 'MSE', 'SSIM'])
  for rec in l:
    table.title = fileName
    table.add_row(rec)
      
  print(table)
  image=Image.open(fileName)
  plt.imshow(image)


# In[ ]:


conciseReportFor('/content/img_1.jpeg')


# In[ ]:


berkeleyImages = ["/content/img_1.jpeg", "/content/img_2.jpeg", "/content/img_3.jpeg",
                  "/content/img_4.jpeg", "/content/img_5.jpeg", "/content/img_6.jpeg",
                  "/content/img_7.jpeg", "/content/img_8.jpeg", "/content/img_9.jpeg",
                  "/content/img_10.jpeg", "/content/img_11.jpeg", "/content/img_12.jpeg",
                  "/content/img_13.jpeg", "/content/img_14.jpeg", "/content/img_15.jpeg"]
for fileName in berkeleyImages:
  conciseReportFor(fileName)

