import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras import Model 
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
import scipy






train_datagen = ImageDataGenerator(rescale = 1.0/255.0,
                                  horizontal_flip = True,
                                  fill_mode = 'nearest',
                                  zoom_range=0.2,
                                  shear_range = 0.2,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  rotation_range=0.4)

train_data = train_datagen.flow_from_directory('preprocess/train',
                                                   batch_size = 5,
                                                   target_size = (350,350),
                                                   class_mode = 'categorical')




val_datagen = ImageDataGenerator(rescale = 1.0/255.0)
val_data = val_datagen.flow_from_directory('preprocess/valid',
                                                   batch_size = 5,
                                                   target_size = (350,350),
                                                   class_mode = 'categorical')




test_datagen = ImageDataGenerator(rescale = 1.0/255.0)
test_data = test_datagen.flow_from_directory('preprocess/test',
                                                   batch_size = 5,
                                                   target_size = (350,350),
                                                   class_mode = 'categorical')


base_model = VGG16(
    weights='imagenet',
    include_top=False, 
    input_shape=(350,350,3)
)




# We define the number of classes in the classification problem.
NUM_CLASSES = 4

# First, a sequential model is created, which will be used to build the VGG model.
vgg_model = Sequential()

# Se agrega una capa al modelo. base_model el modelo anteriormente preentrenado.
vgg_model.add(base_model)

# A flattening layer (Flatten) is added. This layer converts the output from the 
# previous layer (which is likely a three-dimensional tensor) into a one-dimensional vector.
vgg_model.add(layers.Flatten())

# A Dropout layer is added with a dropout rate of 25%. Dropout is used to prevent overfitting 
# by randomly disconnecting some neurons during training.
vgg_model.add(layers.Dropout(0.25))

# A dense layer is added with NUM_CLASSES neurons and a sigmoid activation function. 
# This layer produces the final output of the model.
vgg_model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

# The first layer of the model (base_model) is frozen, so the weights of this layer 
# will not be updated during training.
vgg_model.layers[0].trainable = False

# The model is compiled with the 'categorical_crossentropy' loss function,
#' adam' optimizer, and the accuracy metric. This prepares the model for training.
vgg_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)



mc = ModelCheckpoint(
    filepath="./ct_vgg_best_model.hdf5",
    monitor= 'val_accuracy', 
    verbose= 1,
    save_best_only= True, 
    mode = 'auto'
    );

call_back = [ mc]





vgg = vgg_model.fit(
    train_data, 
    steps_per_epoch = train_data.samples//train_data.batch_size, 
    epochs = 52, 
    validation_data = val_data, 
    validation_steps = val_data.samples//val_data.batch_size,
    callbacks = call_back 
    )






model = load_model("./ct_vgg_best_model.hdf5")




# Checking the Accuracy of the Model 
accuracy_vgg = model.evaluate_generator(generator= test_data)[1] 
print(f"The accuracy of the model is = {accuracy_vgg*100} %")
loss_vgg = model.evaluate_generator(generator= test_data)[0] 
print(f"The loss of the model is = {loss_vgg}")