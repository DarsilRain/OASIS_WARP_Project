#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################################
#------------------------------------------------------------------------------
#******************************************************************************
#Helpful Notes
#
#    Pandas Documentation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
#    Tutorial I'm loosely following: https://keras.io/examples/nlp/multimodal_entailment/
#    ours is a multiclass problem involving four classes: mild dementia, moderate dementia, non demented, and very mild dementia

###############################################################################
#------------------------------------------------------------------------------
#******************************************************************************

# imports
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow import keras


#------------------------------------------------------------------------------
#       Data Preperation
#------------------------------------------------------------------------------
label_map = {'non_demented': 0, 
             'very_mild_dementia': 1,
             'mild_dementia': 2,
             'moderate_dementia' : 3
            }

mild_dementia_path = r'./data/Data/Mild_Dementia'
moderate_dementia_path = r'./data/Data/Moderate_Dementia'
non_demented_path = r'./data/Data/Non_Demented'
very_mild_dementia_path = r'./data/Data/Very_Mild_Dementia'

mild_dementia_list = os.listdir(mild_dementia_path)
moderate_dementia_list = os.listdir(moderate_dementia_path)
non_demented_list = os.listdir(non_demented_path)
very_mild_dementia_list = os.listdir(very_mild_dementia_path)

# add label to each path by putting them in a dictionary
images_dict = {'image_path':[],'image_name':[],'label':[]}

for file in mild_dementia_list:
    images_dict['image_path'].append(mild_dementia_path + '/' + file)
    images_dict['image_name'].append(file)
    images_dict['label'].append('mild_dementia')

for file in moderate_dementia_list:
    images_dict['image_path'].append(moderate_dementia_path + '/' + file)
    images_dict['image_name'].append(file)
    images_dict['label'].append('moderate_dementia')
    
for file in non_demented_list:
    images_dict['image_path'].append(non_demented_path + '/' + file)
    images_dict['image_name'].append(file)
    images_dict['label'].append('non_demented')

for file in very_mild_dementia_list:
    images_dict['image_path'].append(very_mild_dementia_path + '/' + file)
    images_dict['image_name'].append(file)
    images_dict['label'].append('very_mild_dementia')
    
images_df = pd.DataFrame(data=images_dict)

# plot distribution
plt.bar(images_df['label'].unique(),images_df['label'].value_counts())
plt.xticks(rotation=45)

print(images_df['label'].value_counts())

# create stratified split (takes representatives from each category)

images_df['label_idx'] = images_df['label'].apply(lambda x: label_map[x])

# 20% for test
train_df, test_df = train_test_split(images_df, test_size=0.2, 
                                     stratify=images_df['label'].values, 
                                     random_state=42)

#------------------------------------------------------------------------------
#
#       Data Preperation: prepare images
# 
#   How to label images in tf: https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image
#   Flowers Labeling: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb
#
#------------------------------------------------------------------------------
def dataframe_to_dataset(dataframe):
    columns = ["image_path", "label_idx"]
    dataframe = dataframe[columns].copy()
    labels = tf.constant(dataframe['label_idx'].values)
    paths = tf.constant(dataframe['image_path'].values)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

def prepare_dataset(dataframe):
    batch_size = 32
    auto = tf.data.AUTOTUNE
    
    ds = dataframe_to_dataset(dataframe)
    ds = ds.map(lambda x: preprocess_image_from_path(x))
    ds = ds.batch(batch_size).prefetch(auto)
    return ds

# # cleaned and prepared train and test
# train_ds = prepare_dataset(train_df)
test_ds  = prepare_dataset(test_df)

# # View Dataset (DS)
list(test_ds.as_numpy_iterator())
# #list(train_ds.as_numpy_iterator())

for element in test_ds:
    print(type(element))
    print(element)




































