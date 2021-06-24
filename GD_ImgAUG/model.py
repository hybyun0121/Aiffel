import tensorflow.keras as k
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50

class Model():
    def __init__(self, with_mix=False, ds_info=None):
        self.with_mix = with_mix
        self.num_classes = ds_info.features['label'].num_classes
        pass

    def model_setting(self, show_summary=True):
        if not self.with_mix:
            self.resnet = k.models.Sequential(
                [ResNet50(include_top=False,
                          weights='imagenet',
                          input_shape=(224,224,3),
                          pooling='avg',),
                 k.layers.Dense(units=self.num_classes,
                                activation='softmax')])
            if show_summary:
                self.resnet.summary()
            self.resnet.compile(loss='categorical_crossentropy',
                                              optimizer=tf.keras.optimizers.SGD(lr=0.01),
                                              metrics=['accuracy'])

        elif self.with_mix:
            self.resnet = k.models.Sequential(
                [ResNet50(include_top=False,
                          weights='imagenet',
                          input_shape=(224, 224, 3),
                          pooling='avg', ),
                 k.layers.Dense(units=self.num_classes,
                                activation='softmax')])
            if show_summary:
                self.resnet.summary()
            self.resnet.compile(loss='categorical_crossentropy',
                                              optimizer=tf.keras.optimizers.SGD(lr=0.01),
                                              metrics=['accuracy'])
        return self.resnet