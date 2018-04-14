import keras
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(24, (3, 3), input_shape=(1, 50, 50), batch_size=24))
model.compile(optimizer="sgd", loss="categorical_crossentropy")

data_gen = ImageDataGenerator(
                            rotation_range=10,
                              width_shift_range=.1,
                              height_shift_range=.1,
                              shear_range=.02,
                              zoom_range=.05
                              )

data_flow = data_gen.flow_from_directory(directory='D:\\Trevor\\Documents\\_Code\\Python\\Machine Learning\\Sign Language\\SignLanguageProject\\data\\classified\\train',
                                         save_to_dir='D:\\Trevor\\Documents\\_Code\\Python\\Machine Learning\\Sign Language\\SignLanguageProject\\data\\dekart\\trial_gen',
                                         target_size=(150, 150),
                                         batch_size=60,
                                         shuffle=False,
                                         seed=19,
                                         color_mode='grayscale',
                                         save_prefix='gen')

model.fit_generator(data_flow)
