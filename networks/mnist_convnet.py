'''
Author: Trevor Angle
Commented By: Trevor Angle, Krishna Sannasi
'''

'''
This network uses the mnist dataset from Kaggle. The directory strings will need to be updated according to 
where you store the data on your machine.
'''

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import pandas as pd
from sklearn import preprocessing

'''
global parameters
'''
img_width = 28                                              # image parameters
img_height = 28
train_data_directory = '../data/sign_mnist_train.csv'
validation_data_directory = '../data/sign_mnist_test.csv'
batch_size = 64                                             # how many pictures to look at per step
epochs = 10                                                 # how many times to iterate over all pictures

train = pd.read_csv(train_data_directory).values            # getting training data
test = pd.read_csv(validation_data_directory).values        # getting testing data

# coercing data into matricies
if K.image_data_format() == 'channels_first':
    trainX = train[:, 1:].reshape(train.shape[0], 1, 28, 28).astype('float32')
    testX = test[:, 1:].reshape(test.shape[0], 1, 28, 28).astype('float32')
    input_shape = (1, img_width, img_height)
else:
    trainX = train[:, 1:].reshape(train.shape[0], 28, 28, 1).astype('float32')
    testX = test[:, 1:].reshape(test.shape[0], 28, 28, 1).astype('float32')
    input_shape = (img_width, img_height, 1)

# normalizing data
X_train = trainX / 255.0
y_train = train[:, 0]
# y_train /= 255.0

X_test = testX / 255.0
y_test = test[:, 0]
# y_test /= 255.0

# processes the outputs into binary matrices
# this is to allow for easy training classification
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

##############################################

# set up a basic neural network
model = Sequential()

'''
adding layers

architecture:

32 CONV(3x3) - 32 CONV(3x3) - 64 CONV(3x3) - 64 DENSE - 24 DENSE
Inputs         HIDDEN                                   OUTPUT
'''
# input layer
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(.2))

# first hidden
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(.3))

# second hidden
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Dropout(.3))

# third hidden
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(.5))

# output layer
model.add(Dense(24))
model.add(Activation('softmax'))

# construct neural network with above properties
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

##############################################

# train model on training data
model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size
)

model.summary()

# test model on testing data
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Final accuracy on test data: %f.2" % score[1])

# model.save_weights('mnist_try_6.h5')
