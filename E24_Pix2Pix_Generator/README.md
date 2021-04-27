# README

## Pix2Pix 모델을 학습하여 Segmentaion map으로 도로 이미지 만들기

- Data : **`cityscapes`**  

![Screenshot from 2020-11-10 17-43-43](https://user-images.githubusercontent.com/63500940/98650503-53932080-237c-11eb-8b31-ec728ba4918c.png)  

이미지를 반으로 나누어 왼쪽에 있는 Segmentation 이미지를 오른쪽의 도로 이미지로 바꿔주는 모델을 생성한다.

- **Augmentation**  

![image](https://user-images.githubusercontent.com/63500940/98650664-9228db00-237c-11eb-9d9c-15276aa537db.png)
  

데이터를 상하좌우로 랜덤하게 변화를 주어 데이터셋을 더 확보한다.

- **Model**

생성모델로 U-Net을 사용하고 Convolution을 쌓아서 Discriminator를 구현한다.

- **결과( EPOCHS = 100 )**  

![image](https://user-images.githubusercontent.com/63500940/98650833-c56b6a00-237c-11eb-9a3c-b418ec21f7fd.png)
