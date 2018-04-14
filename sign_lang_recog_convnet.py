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

img_width = 150
img_height = 150
train_data_directory = 'data/classified/train'
validation_data_directory = 'data/classified/validation'
data_size = 338
validation_size = 24
batch_size = 24
data_steps_per_epoch = data_size / batch_size
validation_steps_per_epoch = validation_size / batch_size
epochs = 4

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

model = Sequential()

# input layer
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# first hidden
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# second hidden
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# third hidden
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(.01))

# output layer
model.add(Dense(24))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# augmentation config
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=.1,
        height_shift_range=.1,
        shear_range=.02,
        zoom_range=.05
)

# test config
test_datagen = ImageDataGenerator(rescale=1./255)

# train generator
train_generator = train_datagen.flow_from_directory(
    train_data_directory,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    shuffle=True
)

# validation generator
validation_generator = test_datagen.flow_from_directory(
    validation_data_directory,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale'
)

model.fit_generator(
    train_generator,
    steps_per_epoch=data_steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps_per_epoch
)

model.save_weights('try_2.h5')


