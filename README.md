import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
DATA_PATH = '../input/plantdisease/PlantVillage'

for cat in os.listdir(DATA_PATH):
    path = os.path.join(DATA_PATH, cat)
    for img in os.listdir(path):
        image = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
        plt.imshow(image)
        plt.title(f'{cat}')
        plt.show()
        break
IMG_SHAPE = (224, 224)
INPUT_SHAPE = [224, 224, 3]
EPOCHS = 10
BS = 32
img_data_gen = ImageDataGenerator(rescale=1./255, rotation_range=0.2, horizontal_flip=True, vertical_flip=True,
                                 shear_range=0.2, validation_split=0.25)

train_data_gen = img_data_gen.flow_from_directory(DATA_PATH, batch_size=BS, subset='training', 
                                                  class_mode='categorical', shuffle=True) 

val_data_gen = img_data_gen.flow_from_directory(DATA_PATH, batch_size=BS, subset='validation', 
             class_mode='categorical', shuffle=True) 
label = train_data_gen.class_indices
label
img = train_data_gen.__getitem__(11)[0]
plt.imshow(img[0])
plt.figure(figsize=(16,10))
for i in range(15):
    plt.subplot(5, 3, i+1)
    img = train_data_gen.__getitem__(i)[0]
    plt.imshow(img[0])
    plt.xticks()
    plt.show()
    def model_building(model_name, INPUT_SHAPE=INPUT_SHAPE):
    print('Model Initialization started')
    base_model = model_name(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    
for layers in base_model.layers:
        layers.trainable = False
    print('Model Initialization finished')
    
#model creation
    print('Model creation started')
    inp_model = base_model.output
 x = GlobalAveragePooling2D()(inp_model)
    x = Dense(128, activation = 'relu')(x)
    x = Dense(15, activation = 'sigmoid')(x)
    
   model = Model(inputs = base_model.input, outputs = x)
    
   #model summary
    print('Model summary')
    #model.summary()
    
   #model compilation
    model.compile(optimizer = 'adam', metrics=['accuracy'], loss = 'categorical_crossentropy')
      history = model.fit(train_data_gen, validation_data=val_data_gen, 
                       validation_steps=len(val_data_gen)//BS,
                       steps_per_epoch=len(train_data_gen)//BS,
                       batch_size=BS, 
                       epochs=EPOCHS)
    
  print('Model Building Finished')
    
  !mkdir -p saved_model
    model.save(f'saved_model/{model_name}_1.h5')
    print('Model was saved')
    return history
    def evaluation_plot(model):
    sns.set_style('whitegrid')
    
  plt.figure(figsize=(10, 8))
    plt.plot(model['loss'], label = 'loss')
    plt.plot(model['accuracy'], label = 'accuracy')
    plt.plot(model['val_loss'], label = 'val_loss')
    plt.plot(model['val_accuracy'], label = 'val_accuracy')
    plt.legend()
    plt.title('Model Evaluation')
    plt.show()
    from tensorflow.keras.applications.vgg16 import VGG16
    vgg16_hist = model_building(VGG16)
    evaluation_plot(vgg16_hist.history)
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    inc_history = model_building(InceptionV3)
    evaluation_plot(inc_history.history)
