# README

KITTI dataset에서 도로영역 검출 및 Segmentation

U-Net 과 U-Net++ 구현 및 비교

### U-Net++ 도로 영역 Segmentation (IoU : 0.897969)
  
![image](https://user-images.githubusercontent.com/63500940/98757105-204da180-240f-11eb-88a4-3cc2212020c2.png)
  
  
### U-Net 도로 영역 Segmentation (IoU : 0.812884)
  
![image](https://user-images.githubusercontent.com/63500940/98757170-4b37f580-240f-11eb-81bc-ab05a0d629dd.png)
  
  
U-Net++가 인도 영역을 더 잘 구분해내는걸 확인할 수 있다.

### U-Net++ Architecture
  
![image](https://user-images.githubusercontent.com/63500940/98757452-e9c45680-240f-11eb-948b-7c9de9613188.png)
  
$X^{00}, X^{10}, X^{01}$ 부분을 plot 해보면 아래와 같다. 이런식으로 전체 모델을 구현한다.
  
![my_min_model](https://user-images.githubusercontent.com/63500940/98757456-ee890a80-240f-11eb-841e-1271e7abe4bc.png)
  
  
최종 모델의 Plotting 모습
  
![my_final_model](https://user-images.githubusercontent.com/63500940/98757540-15dfd780-2410-11eb-9248-0062c1a34954.png)
  
  
### Loss function
  
a combination of binary cross-entropy and dice coefficient as the loss function:
  
![image](https://user-images.githubusercontent.com/63500940/98757641-563f5580-2410-11eb-97f1-3dda8deeed89.png)
  
  
```python
def my_loss():
    def dice_coef(y_true, y_pred):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f*y_pred_f)
        return (2. * intersection + tf.keras.backend.epsilon()/(tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)+tf.keras.backend.epsilon()))

    def dice_coef_loss(y_true, y_pred):
        return 1-dice_coef(y_true,y_pred)

    def bce_dice_loss(y_true, y_pred):
        loss = tf.keras.losses.binary_crossentropy(y_true,y_pred) + dice_coef_loss(y_true,y_pred)
        return loss
    
    return bce_dice_loss

def bce_dice_loss():
    def f_score(gt, pr, class_weights=1, beta=1, smooth=1, per_image=True, threshold=None):
        if per_image:
            axes = [1, 2]
        else:
            axes = [0, 1, 2]
        if threshold is not None:
            pr = tf.keras.backend.greater(pr, threshold)
            pr = tf.keras.backend.cast(pr, K.floatx())
        tp = tf.keras.backend.sum(gt * pr, axis=axes)
        fp = tf.keras.backend.sum(pr, axis=axes) - tp
        fn = tf.keras.backend.sum(gt, axis=axes) - tp
        score = ((1 + beta ** 2) * tp + smooth) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        # mean per image
        if per_image:
            score = tf.keras.backend.mean(score, axis=0)
        # weighted mean per class
        score = tf.keras.backend.mean(score * class_weights)
        return score
    def dice_loss(gt, pr, class_weights=1., smooth=1, per_image=True, beta=1.):
        return 1 - f_score(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image, beta=beta)
    def bce_dice(gt, pr, bce_weight=1., smooth=1, per_image=True, beta=1.):
        bce = tf.keras.backend.mean(tf.keras.losses.binary_crossentropy(gt, pr))
        loss = bce_weight * bce + dice_loss(gt, pr, smooth=smooth, per_image=per_image, beta=beta)
        return loss
    return bce_dice
```
