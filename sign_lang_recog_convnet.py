'''
This is the actual network. The directory strings will need
to be updated according to where you store the data on your
machine.
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

img_width = 100
img_height = 100
train_data_directory = 'data/classified/train'
validation_data_directory = ''
data_size = 0
validation_size = 0
batch_size = 16
data_steps_per_epoch = data_size / batch_size
validation_steps_per_epoch = validation_size / batch_size
epochs = 50

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
