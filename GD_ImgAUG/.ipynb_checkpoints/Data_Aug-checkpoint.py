import tensorflow as tf
from CutMix import cutmix

import numpy as np
import matplotlib.pyplot as plt

class data_augmentation(cutmix):
    def __init__(self, ds_info):
        self.num_classes = ds_info.features['label'].num_classes

    #이미지 사이즈 통일 및 정규화
    def normalize_and_resize_img(self, image, label):
        """Normalizes images:'unit8'->'float32'"""
        self.image = tf.image.resize(image, [224,224])
        return tf.cast(self.image, tf.float32)/225., label

    #두 가지 변형방법을 사용(좌우 뒤집기, 밝기 변화)
    def augment(self,image,label):
        self.image = tf.image.random_flip_left_right(image)
        self.image = tf.image.random_brightness(self.image, max_delta=0.2)
        return self.image, label

    def onehot(self,image, label):
        self.label = tf.one_hot(label, self.num_classes)
        return image, self.label

    def apply_normalize_on_dataset(self, ds, is_test=False, batch_size=16, with_aug=False, with_cutmix=False):
        self.ds = ds.map(self.normalize_and_resize_img,
                         num_parallel_calls=2)
        if not is_test and with_aug:
            self.ds = self.ds.map(self.augment)

        self.ds = self.ds.batch(batch_size)

        if not is_test and with_cutmix:
            self.ds = self.ds.map(self.cutmix,
                                  num_parallel_calls=2)
        else:
            self.ds = self.ds.map(self.onehot,
                                  num_parallel_calls=2)

        if not is_test:
            self.ds = self.ds.repeat()
            self.ds = self.ds.shuffle(200)
        self.ds = self.ds.prefetch(tf.data.experimental.AUTOTUNE)

        return self.ds


