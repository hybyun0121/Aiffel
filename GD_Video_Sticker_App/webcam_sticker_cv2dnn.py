import numpy as np
import cv2
import dlib
import imutils

from dnnaddsticker import img2sticker_dnn

detector_hog = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

#detection model
prototxt = "./cvdnn/face-detection-with-OpenCV-and-DNN/deploy.prototxt.txt"
caffemodel = "./cvdnn/face-detection-with-OpenCV-and-DNN/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

def main():
    cv2.namedWindow('show', 0)
    cv2.resizeWindow('show', 300, 300)

    vc = cv2.VideoCapture(0)
    #vc = cv2.VideoCapture('./images/video2.mp4')
    img_sticker = cv2.imread('./images/king.png')

    vlen = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print (vlen) # 웹캠은 video length 가 0 입니다.

    # 정해진 길이가 없기 때문에 while 을 주로 사용합니다.
    # for i in range(vlen):
    while True:
        ret, img = vc.read()
        if ret == False:
            break
        start = cv2.getTickCount()
        img = cv2.flip(img, 1)  # 보통 웹캠은 좌우 반전
        img_result = None
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, 
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.5:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([640, 360, 640, 360])
            (startX, startY, endX, endY) = box.astype("int")
            
            # 스티커 메소드를 사용
            img_result = img2sticker_dnn(img, img_sticker.copy(), startX, startY, endX, endY)

        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        print ('[INFO] time: %.2fms'%time)
        
        if img_result is not None:
            cv2.rectangle(img_result, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.imshow('show', img_result)

        else:
            cv2.imshow('show', img)
        
        key = cv2.waitKey(1)
        if key == 27:
            break


if __name__ == '__main__':
    main()
