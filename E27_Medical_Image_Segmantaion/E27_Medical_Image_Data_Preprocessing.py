#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import os
import time
import shutil
import functools

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
from IPython.display import clear_output

import tensorflow as tf
import tensorflow_addons as tfa
print(tf.__version__)


# os.listdir(img_dir) : 해당 경로에서 file.bmp(파일명)을 받아오고  
# img_dir를 붙여서 이미지 파일을 불러온다.

# In[6]:


DATASET_PATH = os.path.join(os.getenv("HOME"),'aiffel/medical')
img_dir = os.path.join(DATASET_PATH, "train")
label_dir = os.path.join(DATASET_PATH, "train_labels")

image_size = 256
img_shape = (image_size, image_size, 3)

def get_train_test_files(img_dir, label_dir):
    x_train_filenames = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]
    x_train_filenames.sort()
    y_train_filenames = [os.path.join(label_dir, filename) for filename in os.listdir(label_dir)]
    y_train_filenames.sort()
    
    #list 형태인 경로 정보를 train_test_split으로 데이터 분할을 한다.
    x_train_filenames, x_test_filenames, y_train_filenames, y_test_filenames = train_test_split(x_train_filenames, y_train_filenames, test_size=0.2)
    num_train_examples = len(x_train_filenames)
    num_test_examples = len(x_test_filenames)

    print(f"# of training examples : {num_train_examples}")
    print(f"# of test examples : {num_test_examples}")
    
    return x_train_filenames, x_test_filenames, y_train_filenames, y_test_filenames, num_train_examples, num_test_examples


# ### Visualization

# #np.random.choice
# ```python
# randnum_int = np.random.randint(0, 100)
# r_choices = np.random.choice(randnum_int, 3)
# print(r_choices)
# ```

# In[7]:


def show_samples(x_train_filenames, y_train_filenames, num_train_examples, display_num):
    r_choices = np.random.choice(num_train_examples, display_num)

    plt.figure(figsize=(10,15)) #전체 plot size 설정
    for i in range(0, display_num*2, 2): #0부터 2칸씩 10까지
        img_num = r_choices[i//2]
        x_pathname = x_train_filenames[img_num]
        y_pathname = y_train_filenames[img_num]

        plt.subplot(display_num, 2, i+1) # 5행 2열에 i+1번째(1,3,5,7,9)
        plt.imshow(Image.open(x_pathname))
        plt.title("Original Image")

        example_labels = Image.open(y_pathname)
        label_vals = np.unique(example_labels)

        plt.subplot(display_num, 2, i + 2)
        plt.imshow(example_labels)
        plt.title("Masked Image")

    plt.suptitle("Example of Images and their Masks")
    plt.show()


# ### Data Preprocessing

# In[10]:


#tf.io.read_file : byte 형태로 데이터를 로드하기
#_img_str = tf.io.read_file('/home/aiffel0042/aiffel/medical/train/71 - Copy.bmp')


# In[11]:


#tf.image.decode_bmp : byte => bmp
#_img = tf.image.decode_bmp(_img_str, channels=3)
#np.shape(_img)


# In[12]:


# 필요한 전처리 기능들을 함수로 정의한다음
# map으로 파일 하나하나에 적용시켜준다.

#데이터 resize & scaling
def _process_pathnames(fname, label_path):
    img_str = tf.io.read_file(fname)
    img = tf.image.decode_bmp(img_str, channels=3)
    
    label_str = tf.io.read_file(label_path)
    label = tf.image.decode_bmp(label_str, channels=1)
    
    resize = [image_size, image_size]
    img = tf.image.resize(img, resize)
    label_img = tf.image.resize(label, resize)
    
    scale = 1/255.
    img = tf.cast(img, dtype=tf.float32) * scale #Casts a tensor to a new type.
    label_img = tf.cast(label_img, dtype=tf.float32) * scale
    
    return img, label_img


# In[13]:


#Data augmentation - Shifting the image
def shift_img(output_img, label_img, width_shift_range, height_shift_range):
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random.uniform([],-width_shift_range*img_shape[1],
                                                 width_shift_range*img_shape[1])
        if height_shift_range:
            height_shift_range = tf.random.uniform([],-height_shift_range*img_shape[0],
                                                  height_shift_range*img_shape[0])
        #tensorflow_addons
        output_img = tfa.image.translate(output_img,
                                        [width_shift_range, height_shift_range])
        label_img = tfa.image.translate(label_img,
                                       [width_shift_range, height_shift_range])
    return output_img, label_img


# In[14]:


#Flipping the image randomly
def flip_img(horizontal_flip, tr_img, label_img):
    if horizontal_flip:
        flip_prob = tf.random.uniform([], 0.0,1.0)
        tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                   lambda: (tf.image.flip_left_right(tr_img),
                                           tf.image.flip_left_right(label_img)),
                                   lambda: (tr_img, label_img))
    return tr_img, label_img


# In[15]:


def _augment(img, label_img, resize=None, scale=1,
             hue_delta=0, horizontal_flip=True,
             width_shift_range=0.05, height_shift_range=0.05):
    if resize is not None:
        label_img = tf.image.resize(label_img, resize)
        img = tf.image.resize(img, resize)
    if hue_delta:
        img = tf.image.random_hue(img, hue_delta)
        
    img, label_img = flip_img(horizontal_flip, img, label_img)
    img, label_img = shift_img(img, label_img, width_shift_range, height_shift_range)
    label_img = tf.cast(label_img, dtype=tf.float32)*scale
    img = tf.cast(img, dtype=tf.float32)*scale
    return img, label_img


# ### Set up train and test datasets

# In[16]:


def get_baseline_dataset(filenames, labels, batch_size, preproc_fn=functools.partial(_augment),
                        threads=4,
                        is_train=True):
    num_x = len(filenames)
    #dataset을 생성한다.
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    #멀티쓰레딩을 주면서 각각의 데이터셋에 preprocessing을 mapping 한다.
    dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
    
    if is_train:
        print("it works")
        dataset = dataset.map(preproc_fn, num_parallel_calls=threads)
        dataset = dataset.shuffle(num_x*10)
    
    dataset = dataset.batch(batch_size)
    return dataset


# In[53]:


#train_dataset = get_baseline_dataset(x_train_filenames, y_train_filenames)
#train_dataset = train_dataset.repeat()

#test_dataset = get_baseline_dataset(x_test_filenames, y_test_filenames, is_train=False)


# #take(1) => batch만큼 가져온다.
# ```python
# for images, labels in train_dataset.take(1):
#     plt.figure(figsize=(10,10))
#     img = images[0]
#     
#     plt.subplot(1,2,1)
#     plt.imshow(img)
#     
#     plt.subplot(1,2,2)
#     plt.imshow(labels[0,:,:,0])
#     plt.show()
# ```

# In[ ]:




