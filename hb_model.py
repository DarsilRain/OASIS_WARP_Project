#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:39:13 2024

@author: haley
"""

###############################################################################
#------------------------------------------------------------------------------
#******************************************************************************
# Helpful Notes
#
#    Pandas Documentation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
#    Tutorial I'm loosely following: https://keras.io/examples/nlp/multimodal_entailment/
#    ours is a multiclass problem involving four classes: mild dementia, moderate dementia, non demented, and very mild dementia

###############################################################################
#------------------------------------------------------------------------------
#******************************************************************************

import tensorflow as tf
#from keras import layers
#from tensorflow.python.keras.layers import Dense, Flatten
#from tensorflow.keras.optimizers import Adam

#------------------------------------------------------------------------------
#       Data Preperation
#
# Tutorial: https://keras.io/examples/vision/image_classification_from_scratch/
#------------------------------------------------------------------------------
image_size = (180, 180)
batch_size = 128

# 20% for testing(validation)
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "./data/Data",
    labels="inferred",
    label_mode='int',
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

#------------------------------------------------------------------------------
#  Build Model
# Tutorial: https://medium.com/@bravinwasike18/building-a-deep-learning-model-with-keras-and-resnet-50-9dd6f4eb3351
# Keras Sequential: https://www.tensorflow.org/guide/keras/sequential_model
#------------------------------------------------------------------------------

dnn_model = tf.keras.Sequential()

# import ResNet50 which is some pre-trained layers from  ImageNet
# classes=4 because of four diagnosis categories
imported_model= tf.keras.applications.ResNet50(include_top=False,
                                               weights='imagenet',
                                               pooling='avg')
for layer in imported_model.layers:
    layer.trainable=False
    
    
dnn_model.add(imported_model)
dnn_model.add(tf.keras.layers.Flatten())
dnn_model.add(tf.keras.layers.Dense(4, activation='softmax')) # 4 classes



# four below because of four diagnosis categories
#dnn_model.add(tf.keras.layers.Dense(4, activation='softmax'))

dnn_model.summary()

dnn_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

history = dnn_model.fit(train_ds,
                        validation_data=val_ds, epochs=10)