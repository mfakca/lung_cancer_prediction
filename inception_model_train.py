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
                                                   class_mode = 'categorical',
                                                   shuffle = False)


# Kullanılacak temel modelin tanımlanması
base_model = InceptionV3(input_shape = (350, 350, 3),
                         include_top = False,
                         weights = 'imagenet')



# Modelin probleme uygun bir şekilde düzenlenmesi
for layer in base_model.layers:
    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(4, activation='sigmoid')(x)

model_incep = tf.keras.models.Model(base_model.input, x)


# Modelde kullanılacak algoritmaların tanımlanması
model_incep.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy',tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])



# En iyi modeli yakalayabilmek için CheckPoint tanımlanması
mc = ModelCheckpoint(
    filepath="./ct_incep_best_model.hdf5",
    monitor= 'val_accuracy',
    verbose= 1,
    save_best_only= True,
    mode = 'auto'
    );

call_back = [mc]


# Modelin eğitilmesi
incep = model_incep.fit(
    train_data,
    steps_per_epoch = train_data.samples//train_data.batch_size,
    epochs = 52,
    validation_data = val_data,
    validation_steps = val_data.samples//val_data.batch_size,
    callbacks = call_back
    )

# En iyi modelin kaydedilmesi
model = load_model("/content/drive/MyDrive/yl_odev/ct_incep_best_model.hdf5")


# Doğrulama skorunun hesaplanması
accuracy_incep = model.evaluate_generator(generator= test_data)[1]
print(f"The accuracy of the model is = {accuracy_incep*100} %")
loss_incep = model.evaluate_generator(generator= test_data)[0]
print(f"The loss the model is = {loss_incep}%")


from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import seaborn as sns

# Test veri seti üzerinde tahminler yapılması
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=-1)

# Gerçek etiketleri alınması
y_true = test_data.classes

# Confusion matrix oluşturulması
conf_matrix = confusion_matrix(y_true, y_pred)

# Confusion matrix'i görselleştirilmesi
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=list(test_data.class_indices.keys()), yticklabels=list(test_data.class_indices.keys()))
plt.title('Confusion Matrix')
plt.ylabel('Gerçek Sınıflar')
plt.xlabel('Tahmin Edilen Sınıflar')
plt.xticks(rotation=45)  
plt.yticks(rotation=45)  

plt.show()




# Precision, Recall ve Accuracy hesaplanması
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')



print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")