import cv2
import numpy as np
import dlib
from tqdm import tqdm
import copy


def img2sticker(img_orig, img_sticker, startX, startY, endX, endY):
    # preprocess
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    # detector
    img_rgb_vga = cv2.resize(img_rgb, (300, 300))
    
    w = (endX - startX)
    h = (endY - startY) // 2
    
    img_sticker = cv2.resize(img_sticker, (w, h), interpolation=cv2.INTER_NEAREST)

    refined_x = startX
    refined_y = startY
    print ('(x,y) : (%d,%d)'%(refined_x, refined_y))

    if refined_y < 0:
        img_sticker = img_sticker[-refined_y:]
        refined_y = 0

    if refined_x < 0:
        img_sticker = img_sticker[:, -refined_x:]
        refined_x = 0
    elif refined_x + img_sticker.shape[1] >= img_orig.shape[1]:
         img_sticker = img_sticker[:, :-(img_sticker.shape[1]+refined_x-img_orig.shape[1])]


    img_bgr = img_orig.copy()
    print("img_bgr : ", np.shape(img_bgr))
    
    sticker_area = img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
    print("img_sticker : " , np.shape(img_sticker))

    img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
         cv2.addWeighted(sticker_area, 1.0, img_sticker, 0.7, 0)

    return img_bgr
