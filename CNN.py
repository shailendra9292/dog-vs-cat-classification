# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout , Flatten
from keras.layers import Conv2D, MaxPooling2D
 
model = Sequential()

# first layer
model.add(Conv2D(128,(3,3), activation='relu', input_shape=(64,64,3)))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))


# second layer
model.add(Conv2D(128,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

#flatten
model.add(Flatten())

#full connection
model.add(Dense(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#compilation
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# data augmentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('/dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

model.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=50,
                    validation_data=test_set ,
                    validation_steps=2000)
