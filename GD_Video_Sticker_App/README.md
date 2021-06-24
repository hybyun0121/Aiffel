# README.md

## 동영상 스티커 앱
  
- 웹캡을 통한 스티커앱 환경 설정  
    - 거리 : 25cm ~ 1m ⇒ 셀카봉 유효거리.
    - 인원 수 : 1명 ⇒ 집에서 혼자 실험할껀데 두 명이상 잡히면 무서우니까
    - 허용 각도 : pitch : -20 ~ 30도, yaw : -45 ~ 45도, roll : -45 ~ 45도 → 화면을 바라볼 수 있는 각도
    - 안정성 : 안정성 : 위 조건을 만족하면서 FPPI (false positive per image) 기준 < 0.003, MR (miss rate) < 1 300장당 1번 에러 = 10초=30*10에 1번 에러

    ⇒ 기준의 이유 : 다양한 조건을 실험하기에 환경적 제약이 많고, 시간적 여유도 적었다. 그리고 예시로 들어준 조건들이 타당해 보여 참고하였다.  
  
- 프레임 좌표범위 예외처리 관련 오류 수정  
아래와 같이 위로 벗어나거나 양옆으로 벗어나는 예외의 경우 오류가 발생하지않게 처리하였다.  
  
    ```python
    if refined_y < 0:
            img_sticker = img_sticker[-refined_y:]
            refined_y = 0

        if refined_x < 0:
            img_sticker = img_sticker[:, -refined_x:]
            refined_x = 0
        elif refined_x + img_sticker.shape[1] >= img_orig.shape[1]:
             img_sticker = img_sticker[:, :-(img_sticker.shape[1]+refined_x-img_orig.shape[1])]
    ```
  
- 스티커앱 안정성 여부 확인하기  
칼만필터를 적용하였을때 안정적으로 스티커가 잘 붙어있는걸 확인할 수 있어지만,
속도가 느리고 프레임이 끊기는 듯한 영상을 확인할 수 있었다.
해결 방법으로 openCV에서 제공하는 dnn 라이브러리를 활용하여 얼굴 영역을 검출 하고 bbox 좌표 정보를 활용하여 구현하였다.
속도도 훨씬 빠르고 스티커 앱도 끊김없이 사람의 얼굴의 이동을 따라 부드럽게 출력되는 것을 확인할 수 있었다.  
  
![result_crop](https://user-images.githubusercontent.com/63500940/100684372-92395b00-33bd-11eb-98f4-765b0274b2e9.png)  
  
[참고] https://github.com/sr6033/face-detection-with-OpenCV-and-DNN
