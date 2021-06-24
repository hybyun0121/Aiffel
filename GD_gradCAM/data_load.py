import os
import copy
import cv2
import urllib3
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

class DataLoad():
    def __init__(self,):
        pass

    def load_tfds(self):
        urllib3.disable_warnings()
        (ds_train, ds_test), ds_info = tfds.load(
            'cars196',
            split=['train', 'test'],
            shuffle_files=True,
            with_info=True,
        )
        return (ds_train, ds_test), ds_info

    def normalize_and_resize_img(self, input):
        image = tf.image.resize(input["image"], [224,224])
        input['image']=tf.cast(image, tf.float32)/255.
        return input["image"], input["label"]

    def apply_normalize_on_dataset(self, ds, is_test=False, batch_size=16):
        ds = ds.map(
            self.normalize_and_resize_img,
            num_parallel_calls=2
        )
        ds = ds.batch(batch_size)
        if not is_test:
            ds = ds.repeat()
            ds = ds.shuffle(200)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds