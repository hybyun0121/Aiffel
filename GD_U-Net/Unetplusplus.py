import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras import Model
'''
참고사이트
https://paperswithcode.com/paper/unet-a-nested-u-net-architecture-for-medical
'''
class Convblock(tf.keras.Model):
    def __init__(self, filter1, filter2):
        super(Convblock,self).__init__()
        self.act = tf.keras.layers.Activation(tf.nn.relu)
        self.conv1 = tf.keras.layers.Conv2D(filters=filter1,kernel_size=3,
                                            strides=1,padding="same", kernel_initializer="he_normal")
        self.BN1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter2,kernel_size=3,
                                            strides=1, padding='same', kernel_initializer="he_normal")
        self.BN2 = tf.keras.layers.BatchNormalization()

    def call(self, input):
        x = self.conv1(input)
        x = self.BN1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.BN2(x)
        return self.act(x)


class UNETPP(tf.keras.Model):
    def __init__(self, first_filter=32):
        super(UNETPP, self).__init__()
        filters = []
        for i in range(5):
            filters.append(2**i * first_filter)

        self.maxpooling = tf.keras.layers.MaxPooling2D()
        self.up = tf.keras.layers.UpSampling2D()

        self.conv0_0 = Convblock(filter1=filters[0], filter2=filters[0])
        self.conv1_0 = Convblock(filter1=filters[0], filter2=filters[1])
        self.conv2_0 = Convblock(filter1=filters[1], filter2=filters[2])
        self.conv3_0 = Convblock(filter1=filters[2], filter2=filters[3])
        self.conv4_0 = Convblock(filter1=filters[3], filter2=filters[4])

        self.conv0_1 = Convblock(filter1=filters[0], filter2=filters[0])
        self.conv0_2 = Convblock(filter1=filters[0], filter2=filters[0])
        self.conv0_3 = Convblock(filter1=filters[0], filter2=filters[0])

        self.conv1_1 = Convblock(filter1=filters[1], filter2=filters[1])
        self.conv1_2 = Convblock(filter1=filters[1], filter2=filters[1])

        self.conv2_1 = Convblock(filter1=filters[2], filter2=filters[2])

        self.conv3_1 = Convblock(filter1=filters[3], filter2=filters[3])
        self.conv2_2 = Convblock(filter1=filters[2], filter2=filters[2])
        self.conv1_3 = Convblock(filter1=filters[1], filter2=filters[1])
        self.conv0_4 = Convblock(filter1=filters[0], filter2=filters[0])

        self.activation1 = tf.keras.layers.Activation(tf.nn.relu)
        self.activation2 = tf.keras.layers.Activation(tf.nn.relu)
        self.activation3 = tf.keras.layers.Activation(tf.nn.relu)
        self.conv_last1 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=3,
                                                padding='same',
                                                kernel_initializer="he_normal")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv_last2 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=3,
                                                padding='same',
                                                kernel_initializer="he_normal")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv_last3 = tf.keras.layers.Conv2D(filters=2,
                                                kernel_size=3,
                                                padding='same',
                                                kernel_initializer="he_normal")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv_last4 = tf.keras.layers.Conv2D(filters=1,kernel_size=1,activation='sigmoid')


    def call(self, inputs, training=False):
        x0_0 = self.conv0_0(inputs)
        x1_0 = self.conv1_0(self.maxpooling(x0_0))
        x2_0 = self.conv2_0(self.maxpooling(x1_0))
        x3_0 = self.conv3_0(self.maxpooling(x2_0))
        x4_0 = self.conv4_0(self.maxpooling(x3_0))

        x0_1 = self.conv0_1(Concatenate()([x0_0, self.up(x1_0)]))
        x1_1 = self.conv1_1(Concatenate()([x1_0, self.up(x2_0)]))
        x2_1 = self.conv2_1(Concatenate()([x2_0, self.up(x3_0)]))
        x3_1 = self.conv3_1(Concatenate()([x3_0, self.up(x4_0)]))
        x2_2 = self.conv2_2(Concatenate()([x2_0,x2_1,self.up(x3_1)]))

        x0_2 = self.conv0_2(Concatenate()([x0_0,x0_1,self.up(x1_1)]))
        x1_2 = self.conv1_2(Concatenate()([x1_0,x1_1,self.up(x2_1)]))
        x1_3 = self.conv1_3(Concatenate()([x1_0,x1_1,x1_2,self.up(x2_2)]))

        x0_3 = self.conv0_3(Concatenate()([x0_0,x0_1,x0_2,self.up(x1_2)]))

        x0_4 = self.conv0_4(Concatenate()([x0_0,x0_1,x0_2,x0_3,self.up(x1_3)]))
        output = self.conv_last1(x0_4)
        output = self.bn1(output)
        output = self.activation1(output)
        output = self.conv_last2(output)
        output = self.bn2(output)
        output = self.activation2(output)
        output = self.conv_last3(output)
        output = self.bn3(output)
        output = self.activation3(output)
        output = self.conv_last4(output)
        return output

    def get_summary(self, input_shape=(224,224,3), plotting=False):
        inputs = Input(input_shape)
        _model = Model(inputs, self.call(inputs))
        if plotting:
            tf.keras.utils.plot_model(_model, to_file="./my_final_model.png", show_shapes=True)
        return _model.summary()