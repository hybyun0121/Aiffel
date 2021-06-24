import tensorflow as tf

import numpy as np
import tensorflow_datasets

class cutmix():
    def __init__(self):
        pass

    # 섞을 두 이미지의 바탕이미지에 삽입할 두번째 이미지가 있을 때,
    # 바탕이미지에 삽입될 영역의 바운딩 박스의 위치를 결정하는 함수
    def get_clip_box(self, iamge_a, image_b, img_size=224):
        # get center of box
        self.x = tf.cast(tf.random.uniform([], 0, img_size), tf.int32)
        self.y = tf.cast(tf.random.uniform([], 0, img_size), tf.int32)

        # get width of box
        self._prob = tf.random.uniform([], 0, 1)
        self.width = tf.cast(img_size * tf.math.sqrt(1 - self._prob), tf.int32)

        # clip box in image and get minmax bbox
        self.xa = tf.math.maximum(0, self.x - self.width // 2)
        self.ya = tf.math.maximum(0, self.y - self.width // 2)
        self.yb = tf.math.minimum(img_size, self.y + self.width // 2)
        self.xb = tf.math.minimum(img_size, self.x + self.width // 2)

        return self.xa, self.ya, self.xb, self.yb

    # mix two images
    def mix_2_images(self,image_a, image_b, xa, ya, xb, yb, img_size=224):
        self.one = image_a[ya:yb, 0:xa, :]
        self.two = image_b[ya:yb, xa:xb, :]
        self.three = image_a[ya:yb, xb:img_size, :]
        self.middle = tf.concat([self.one, self.two, self.three], axis=1)
        self.top = image_a[0:ya, :, :]
        self.bottom = image_a[yb:img_size, :, :]
        self.mixed_img = tf.concat([self.top, self.middle, self.bottom], axis=0)

        return self.mixed_img

    # mix two labels
    def mix_2_label(self, label_a, label_b, xa, ya, xb, yb, img_size=224, num_classes=120):
        self.mixed_area = (xb - xa) * (yb - ya)
        self.total_area = img_size * img_size
        self.a = tf.cast(self.mixed_area / self.total_area, tf.float32)  # 섞인 비율

        if len(label_a.shape) == 0:
            self.label_a = tf.one_hot(label_a, num_classes)
        if len(label_b.shape) == 0:
            self.label_b = tf.one_hot(label_b, num_classes)
        self.mixed_label = (1 - self.a) * self.label_a + self.a * self.label_b
        return self.mixed_label

    # 위에서 구현한 함수로 cutmix() 함수 완성하기
    def cutmix(self, image, label, prob=1.0, batch_size=16, img_size=224, num_classes=120):
        self.mixed_imgs = []
        self.mixed_labels = []

        for i in range(batch_size):
            self.image_a = image[i]  # 바탕 이미지
            self.label_a = label[i]
            self.j = tf.cast(tf.random.uniform([], 0, batch_size), tf.int32)
            self.image_b = image[self.j]  # 랜덤하게 뽑힌 crop될 이미지
            self.label_b = label[self.j]
            self.xa, self.ya, self.xb, self.yb = self.get_clip_box(self.image_a, self.image_b)
            self.mixed_imgs.append(self.mix_2_images(self.image_a, self.image_b, self.xa, self.ya, self.xb, self.yb))
            self.mixed_labels.append(self.mix_2_label(self.label_a, self.label_b, self.xa, self.ya, self.xb, self.yb))

        self.mixed_imgs = tf.reshape(tf.stack(self.mixed_imgs), (batch_size, img_size, img_size, 3))
        self.mixed_labels = tf.reshape(tf.stack(self.mixed_labels), (batch_size, num_classes))
        return self.mixed_imgs, self.mixed_labels


