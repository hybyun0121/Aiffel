import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

# 모델을 구현하기 위해 필요한 block들을 정의한다.
class ResNetLayerblocks(tf.keras.Model):
    def __init__(self):
        super(ResNetLayerblocks, self).__init__()
    
    def head_block(self, input_layer):
        '''
            input:
                input_layer : (BATCH_SIZE, Height, Width, Channel)
            output:
                x : head_block을 통과한 tensor로 공간적 크기는 줄고 channel은 증가한다.
        '''
        x = input_layer
        x = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2,padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
        return x
    
    def residual_conv_block(self, input_layer, channel, filter_type=1, on_act=True):
        '''
            Args:
                input_layer : input으로 들어오는 data
                channel : Conv layer에서 출력으로 만들 channel size
                filter_type : int로 1이면 1x1 conv, 3이면 3x3 conv를 수행한다.
                on_act : bool type으로 True이면 Conv->BN->ReLU이고 False이면 ReLU를 수행하지 않는다.
        '''
        x = input_layer
        x = keras.layers.Conv2D(channel, (filter_type,filter_type), strides=1, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        
        if on_act:
            x = keras.layers.ReLU()(x)
        return x
    
    def bottleneck_block(self, input_layer, channel, num_iter=3, add_id = True):
        '''
            Args:
                input_layer : Skip connection을 타고 이동할 Tensor
                channel : block에서 conv layer들이 계산할 channel size
                num_iter : 하나의 bottleneck block을 반복할 횟수
                add_id : bool type으로 True이면 skip connection을 False이면 일반적인 convloution layer가
                         만들어진다.
        '''
        f_x = x_id = input_layer # identitiy X와 F(x)를 구분하기 위해 두 변수에 나눠 담는다.
        
        # bottlenect block을 구성하는 작은 block들을 쌓아준다.
        for _ in range(num_iter):
            f_x = self.residual_conv_block(f_x, channel, filter_type=1, on_act=True)
            f_x = self.residual_conv_block(f_x, channel, filter_type=3, on_act=True)

            # 마지막 레이어는 skipped connection을 할 경우를 위해 ReLU을 따로 빼서 계산한다. 
            f_x = self.residual_conv_block(f_x, channel*4, filter_type=1, on_act=False)

            if add_id:
                # identitiy X의 channel을 맞춰주기 위해 1x1 Conv에 태워준다.
                # x_id = keras.layers.Conv2D(channel*4, kernel_size=(1,1),strides=1,padding='same')(x_id)
                # x_id = keras.layers.BatchNormalization()(x_id)
                x_id = self.zero_padding(x_id, channel*4)

                f_x = keras.layers.Add()([f_x, x_id]) # F(x)+x 부분
                f_x = keras.layers.ReLU()(f_x)

                x_id = f_x
            else:
                f_x = keras.layers.ReLU()(f_x)
                x_id = f_x
        
        return f_x
    
    def zero_padding(self, x_id, channel):
        diff = channel - x_id.shape[3]
        padding = [[0, 0], [0, 0], [0, 0], [0, diff]]
        x_id = tf.cast(x_id, tf.int64)
        x_id = tf.pad(x_id, padding)
        return x_id
        
        
    def last_layer(self, input_layer, num_class):
        '''
            Args:
                input_layer : 마지막 추론단으로 들어오는 Tensor
                num_class : Classification하고자 하는 label 종류
        '''
        x = input_layer
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(num_class)(x)
        return x

class Resnet50(ResNetLayerblocks):
    def __init__(self, input_shape, add_id, num_blocks=[], num_class=10):
        '''
            Args:
                input_shape : 학습데이터의 image shape
                add_id : bool type. Skip connection의 유무.
                num_blocks : list type. 큰 단위의 블럭으로 bottleneck block의 갯수를 알려준다.
                num_class : class의 수.
        '''
        super(Resnet50, self).__init__()
        self.Input = keras.layers.Input(input_shape)
        self.num_blocks = num_blocks
        self.add_id = add_id
        self.num_class = num_class
        
    def build_model(self):
        '''
        Model을 생성하는 함수이다.
        head_block -> bottleneck_block x n -> last_layer 순서로
        layer들이 쌓이게된다.
        
        bottleneck_block이 n개 쌓이게 되고 각각의 block 안에는
        1x1 conv와 3x3 conv들이 쌓여서 작은 단위의 block을 이루고 있다.
        '''
        x = self.head_block(self.Input)
        for i in range(len(self.num_blocks)):
            if i == 0:
                x = self.bottleneck_block(x,channel = x.shape[-1], num_iter=self.num_blocks[i],add_id=self.add_id)
            else:
                x = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
                x = self.bottleneck_block(x, channel = (x.shape[-1]/2), num_iter=self.num_blocks[i],add_id=self.add_id)
        
        out = self.last_layer(x,num_class=self.num_class)
        model = keras.Model(inputs=self.Input, outputs=out)
        return model
    
    def __call__(self):
        print("모델 생성 완료")
        return self.build_model()


