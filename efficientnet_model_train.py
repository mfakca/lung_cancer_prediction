# Kullanılan kütüphanelerin içeri aktarılması
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

base_path = '/content/drive/MyDrive/yl_odev/'


# Eğitim datası için DataGenerator objesi oluşturması
train_datagen = ImageDataGenerator(rescale = 1.0/255.0,
                                  horizontal_flip = True,
                                  fill_mode = 'nearest',
                                  zoom_range=0.2,
                                  shear_range = 0.2,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  rotation_range=0.4)
# Eğitim datasının özelliklerinin tanımlanması
train_data = train_datagen.flow_from_directory(base_path + 'preprocess/train',
                                                   batch_size = 5,
                                                   target_size = (350,350),
                                                   class_mode = 'categorical')




# Doğrulama datası için DataGenerator objesi oluşturması
val_datagen = ImageDataGenerator(rescale = 1.0/255.0)
# Doğrulama datasının özelliklerinin tanımlanması
val_data = val_datagen.flow_from_directory(base_path + 'preprocess/valid',
                                                   batch_size = 5,
                                                   target_size = (350,350),
                                                   class_mode = 'categorical')



# Test datası için DataGenerator objesi oluşturması
test_datagen = ImageDataGenerator(rescale = 1.0/255.0)

# Test datasının özelliklerinin tanımlanması
test_data = test_datagen.flow_from_directory(base_path + 'preprocess/test',
                                                   batch_size = 5,
                                                   target_size = (350,350),
                                                   class_mode = 'categorical')


# Kullanılacak temel modelin tanımlanması
base_model = InceptionV3(input_shape = (350, 350, 3),
                         include_top = False,
                         weights = 'imagenet')


# En iyi modeli yakalayabilmek için CheckPoint tanımlanması
mc = ModelCheckpoint(
    filepath="/content/drive/MyDrive/yl_odev/ct_effnet_best_model.hdf5",
    monitor= 'val_accuracy',
    verbose= 1,
    save_best_only= True,
    mode = 'auto'
    );

call_back = [ mc];

tensorboard = TensorBoard(log_dir = 'logs')
reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.3, patience = 2, min_delta = 0.001,
                              mode='auto',verbose=1)




early_stopping = EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True)

# Modelin probleme uygun bir şekilde düzenlenmesi
EffNetmodel = base_model.output
EffNetmodel = tf.keras.layers.GlobalAveragePooling2D()(EffNetmodel)
EffNetmodel = tf.keras.layers.Dropout(rate=0.25)(EffNetmodel)
EffNetmodel = tf.keras.layers.Dense(4,activation='softmax')(EffNetmodel)
EffNetmodel = tf.keras.models.Model(inputs=base_model.input, outputs = EffNetmodel)



# Modelde kullanılacak algoritmaların tanımlanması
EffNetmodel.compile(loss='categorical_crossentropy',optimizer = 'Adam', metrics= ['accuracy'])



# Modelin eğitilmesi
EffNetB0 = EffNetmodel.fit(
    train_data,
    steps_per_epoch = train_data.samples//train_data.batch_size,
    epochs = 52,
    validation_data = val_data,
    validation_steps = val_data.samples//val_data.batch_size,
    callbacks = [tensorboard, mc, reduce_lr, early_stopping]
    )


# En iyi modelin kaydedilmesi
model_eff = load_model("/content/drive/MyDrive/yl_odev/ct_effnet_best_model.hdf5")


# Doğrulama skorunun hesaplanması
accuracy_effnet = model_eff.evaluate_generator(generator= test_data)[1]
loss_effnet = model_eff.evaluate_generator(generator= test_data)[0]
print(f"The accuracy of the model is = {accuracy_effnet*100} %")
print(f"The loss of the model is = {loss_effnet} %")