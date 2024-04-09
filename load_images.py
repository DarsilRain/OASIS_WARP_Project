#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:15:16 2024

@author: haley

# Tutorial: https://keras.io/examples/vision/image_classification_from_scratch/
"""

import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

num_skipped = 0
for folder_name in ("Mild_Dementia", "Moderate_Dementia", "Non_Demented","Very_Mild_Dementia"):
    folder_path = os.path.join("./data/Data", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print(f"Deleted {num_skipped} images.")

image_size = (180, 180)
batch_size = 128

# 20% for testing(validation)
train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "./data/Data",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

