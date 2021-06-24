import os
from functools import partial


import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import urllib3
urllib3.disable_warnings()

class Data_load():
    def __init__(self, data_name='cifar10', file_path=None):
        self.data_name = data_name
        self.file_path = file_path

    def get_tfds(self, show_info=True):
        '''
        :param show_info:
        :return: (ds_train, ds_test), ds_info
        '''
        if not self.file_path == None:
            (self.ds_train, self.ds_test), self.ds_info = tfds.load(self.data_name,
                                                                    split=['train','test'],
                                                                    shuffle_files=True,
                                                                    with_info=True,
                                                                    as_supervised=True,
                                                                    data_dir=self.file_path)
        else:
            (self.ds_train, self.ds_test), self.ds_info = tfds.load(self.data_name,
                                                                    split=['train','test'],
                                                                    with_info=True,
                                                                    shuffle_files=True,
                                                                    as_supervised=True)
        if show_info == True:
            print(f'데이터 특성 : {self.ds_info.features}')
            print(f'Train data : {tf.data.experimental.cardinality(self.ds_train)}')
            print(f'Test data : {tf.data.experimental.cardinality(self.ds_test)}')

        return (self.ds_train, self.ds_test), self.ds_info

    def _normalize_and_resize_img(self, img, label, image_size=32):
        image = tf.image.resize(img, [image_size, image_size])
        return ((tf.cast(image, tf.float32)/128. -1), label)

    def apply_normalize_on_dataset(self, ds, is_train=True, image_size = 32, batch_size=16):        
        # tfds 데이터 타입 관리가 헷갈린다...
        ds = ds.map(partial(self._normalize_and_resize_img, image_size=image_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        if not is_train:
            ds = ds.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
        else:
            ds = ds.repeat().shuffle(self.ds_info.splits['train'].num_examples).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
        return ds