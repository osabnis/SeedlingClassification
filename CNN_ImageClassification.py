# PLANT IMAGE CLASSIFICATION USING CNN
#By: Omkar Sabnis - 24.05.2018

# IMPORTING ALL NECESAARY MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math
from glob import glob
import itertools

# KERAS AND SKLEARN
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,CSVLogger
from sklearn.metrics import confusion_matrix

