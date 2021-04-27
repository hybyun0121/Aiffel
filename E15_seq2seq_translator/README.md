## seq2seq : English to French Translator

### Word-Level로 단어 Embedding하기

영어 문장을 입력 받아서 프랑스어 문장으로 번역하기 위해 시퀀스 모델 중 many to many 방식으로 모델을 구성한다.

![jZfab](https://user-images.githubusercontent.com/63500940/94544605-6814d100-0286-11eb-82dc-77a4c7506de9.jpg)


### Encoding & Decoding

Encoder : 영어 문장을 입력받아 feature를 뽑아내는 역할을 한다.

Decoder : Encoder에서 뽑아낸 feature를 입력받아 프랑스어로 출력하는 역할을 한다.

우리의 목표는 Encoder와 Decoder의 성능을 높여 영어문장을 프랑스어로 잘 번역하도록 하는 것이다.

### Tokenizer

word 단위로 문장을 쪼개어 단어마다 고유한 번호를 부여한다. 즉 순서대로 나열하고 인덱스 번호를 부여한다. 이는 모델에 문자열 데이터가 아닌 숫자형 데이터로 입력을 하기 위함이다. 토크나이징을 하면 word 단위로 매칭되는 정수들의 나열로 한 문장을 표현할 수 있다.

### Embedding

단어들을 임의의 차원의 크기를 가지는 공간에 벡터로 표현한다. 

word들 사이에 상관관계를 반영할 수 있다.

![VisualizeWordEmbeddingsUsingTextScatterPlotsExample_01](https://user-images.githubusercontent.com/63500940/94544684-811d8200-0286-11eb-8618-8c3f4529be11.png)

위 그림을 보면 각각의 점들이 하나의 단어를 뜻한다.

### Teacher forcing

![image](https://user-images.githubusercontent.com/63500940/94544749-9692ac00-0286-11eb-8af6-2a9841c8c6ac.png)


매 스텝에서 나온 출력을 다음 스텝의 입력으로 주는 것이 아니라 정답 정보를 직접 입력함으로써 학습의 정확도를 높이는 방법이다. 학습데이터에대한 정확도는 증가하겠지만 창의적인 문장을 생성하는 것에는 제약을 준다.

### Callbacks

효율적인 모델 학습을 위하여 세 가지 callback을 생성한다.

1. model_checkpoint_callback : `val_acc` 를 모니터링하면서 최댓값을 가지는 모델의 파라미터를 저장한다.
2. earlystopping : `val_acc` 가 10번동안 변화가 없으면 학습을 중단한다.
3. scheduler_callback : `CosineAnnealingScheduler` 를 사용하여 에폭마다 learning rate에 변화를 준다.

![download (1)](https://user-images.githubusercontent.com/63500940/94545128-1ae52f00-0287-11eb-9e9d-bb0221efac8e.png)


### 예측 번역 문장

```
원문 :  they spoke briefly . 
번역문 : ils se sont bri vement entretenus . 
예측문 :  elles parl rent bri vement . 

원문 :  i got really mad . 
번역문 : je me suis vraiment mise en col re . 
예측문 :  je me suis vraiment mis en col re . 

원문 :  i wonder who . 
번역문 : je me demande qui . 
예측문 :  je me demande qui . 

원문 :  it was very ugly . 
번역문 : c tait tr s laid . 
예측문 :  c tait fort laid . 

원문 :  she became pregnant . 
번역문 : elle est tomb e enceinte . 
예측문 :  elle est tomb e enceinte . 

원문 :  i ve just seen tom . 
번역문 : je viens de voir tom . 
예측문 :  je viens de voir tom . 

원문 :  she met her uncle . 
번역문 : elle rencontra son oncle . 
예측문 :  elle a rencontr son oncle . 
```

전반적인 모델 구현은 아래 블로그를 참고하였다.

[Neural Machine Translation (seq2seq) Tutorial](https://wikidocs.net/86900)

사진출처
[https://stackoverflow.com/questions/47594861/predicting-a-multiple-forward-time-step-of-a-time-series-using-lstm](https://stackoverflow.com/questions/47594861/predicting-a-multiple-forward-time-step-of-a-time-series-using-lstm)
[https://kr.mathworks.com/help/textanalytics/ug/visualize-word-embedding-using-text-scatter-plot.html](https://kr.mathworks.com/help/textanalytics/ug/visualize-word-embedding-using-text-scatter-plot.html)
[https://blog.naver.com/PostView.nhn?blogId=sooftware&logNo=221790750668&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView](https://blog.naver.com/PostView.nhn?blogId=sooftware&logNo=221790750668&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView)


