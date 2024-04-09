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

# 5% for validation
train_df, val_df = train_test_split(
    train_df, test_size=0.05, stratify=train_df["label"].values, random_state=42
)

print(f"Total training examples: {len(train_df)}")
print(f"Total validation examples: {len(val_df)}")
print(f"Total test examples: {len(test_df)}")

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

def preprocess_image_from_path(image_path):
    
    resize = (128, 128)
        
    try: 
        image_string = tf.io.read_file(image_path)
        image_decoded = tf.image.decode_jpeg(image_string, 3)
        image = tf.image.resize(image_decoded, resize)
    except:
        print("Error Processing Image")
        
    return image


def prepare_dataset(dataframe):
    batch_size = 32
    auto = tf.data.AUTOTUNE
    
    ds = dataframe_to_dataset(dataframe)
    ds = ds.map(lambda x, y: preprocess_image_from_path(x))
    ds = ds.batch(batch_size).prefetch(auto)
    return ds

# # cleaned and prepared train and test
train_ds = prepare_dataset(train_df)
test_ds  = prepare_dataset(test_df)
validation_ds = prepare_dataset(val_df)

# # # View Dataset (DS)
# list(test_ds.as_numpy_iterator())
# # #list(train_ds.as_numpy_iterator())

# for element in test_ds:
#     print(type(element))
#     print(element)


#------------------------------------------------------------------------------
#  Build Model
#------------------------------------------------------------------------------

def project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
):
    
    projected_embeddings = keras.layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = keras.layers.Dense(projection_dims)(x)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Add()([projected_embeddings, x])
        projected_embeddings = keras.layers.LayerNormalization()(x)
    return projected_embeddings

def create_vision_encoder(
    num_projection_layers, projection_dims, dropout_rate, trainable=False
):
    # Load the pre-trained ResNet50V2 model to be used as the base encoder.
    resnet_v2 = keras.applications.ResNet50V2(
        include_top=False, weights="imagenet", pooling="avg"
    )
    # Set the trainability of the base encoder.
    for layer in resnet_v2.layers:
        layer.trainable = trainable

    # Receive the images as inputs.
    image_1 = keras.Input(shape=(128, 128, 3), name="image_1")

    # Preprocess the input image.
    preprocessed_1 = keras.applications.resnet_v2.preprocess_input(image_1)

    # Generate the embeddings for the images using the resnet_v2 model
    # concatenate them.
    embeddings = resnet_v2(preprocessed_1)

    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the vision encoder model.
    return keras.Model(image_1, outputs, name="vision_encoder")

def create_model(
    num_projection_layers=1,
    projection_dims=256,
    dropout_rate=0.1,
    vision_trainable=False,
    text_trainable=False,
):
    # Receive the images as inputs.
    image_1 = keras.Input(shape=(128, 128, 3), name="image_1")

    # Create the encoders.
    vision_encoder = create_vision_encoder(
        num_projection_layers, projection_dims, dropout_rate, vision_trainable
    )

    # Fetch the embedding projections.
    vision_projections = vision_encoder(image_1)

    # pass through the classification layer.
    outputs = keras.layers.Dense(3, activation="softmax")(vision_projections)
    return keras.Model(image_1, outputs)


model = create_model()
keras.utils.plot_model(model, show_shapes=True)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy"
)

history = model.fit(train_ds, validation_data=validation_ds, epochs=10)


#------------------------------------------------------------------------------
#  Evaluate Model
#------------------------------------------------------------------------------
_, acc = model.evaluate(test_ds)
print(f"Accuracy on the test set: {round(acc * 100, 2)}%.")


























