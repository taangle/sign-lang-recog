import keras
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(24, (3, 3), input_shape=(50, 50, 1), batch_size=24))
model.compile(optimizer="sgd", loss="categorical_crossentropy")

data_gen = ImageDataGenerator(rotation_range=15,
                              width_shift_range=15,
                              height_shift_range=15)

data_flow = data_gen.flow_from_directory(directory='D:\\Trevor\\Documents\\_Code\\Python\\Machine Learning\\Sign Language\\data\\signingsavvy\\letters',
                                         save_to_dir='D:\\Trevor\\Documents\\_Code\\Python\\Machine Learning\\Sign Language\\data\\signingsavvy\\trial_2',
                                         target_size=(100, 100),
                                         batch_size=24,
                                         shuffle=False,
                                         seed=19,
                                         color_mode='grayscale',
                                         save_prefix='gen')

model.fit_generator(data_flow)
