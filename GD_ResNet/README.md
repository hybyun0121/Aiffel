# README

ResNet 구현해보기

[논문링크]  [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

논문에서 소개한 방법을 최대한

존중하고 상세히 설명되지 않은 부분은 직접 채워 넣었다.

Architecture Idea

![Untitled](https://user-images.githubusercontent.com/63500940/96157140-e0e57f80-0f4c-11eb-89da-07e521151261.png)
(README%20aa143dfe50a54d6d9dee6933112a288e/Untitled.png)

ResNet-34는 총 34개의 layer로 구성되어 있다.

첫 블록은 단일 Convoultion(conv) layer이며 이후 Max pooling이 이어진다.

> We adopt batch-normalization (BN) [16] right after each convolution and before activation, following [16].

논문에서 Conv layer 이후 BN을 통과시킨다음 activation function을 적용시킨다고 하였다.

따라서 모든 Conv layer 뒤에는 BN 다음 activation layer가 오도록 한다.

![Untitled 1](https://user-images.githubusercontent.com/63500940/96157192-f490e600-0f4c-11eb-8f40-bf6013bed6c2.png)
(README%20aa143dfe50a54d6d9dee6933112a288e/Untitled%201.png)

> The dotted shortcuts increase dimensions. Table 1 shows more details and other variants.

논문에서 새로운 블록으로 feature map이 입력될때 채널의 사이즈는 두배로 증가하고 resoultion은 1/2배 감소하기 때문에 identity mapping을 하기 위해선 dotted shortchuts의 사이즈를 변화해줘야한다.

> When the dimensions increase (dotted line shortcuts in Fig. 3), we consider two options: (A) The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions. This option introduces no extra
parameter; (B) The projection shortcut in Eqn.(2) is used to match dimensions (done by 1×1 convolutions). For both options, when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2.

이때 두가지 방법이 소개되는데 일단 채널의 차원을 늘리는 방법에 대한 접근이다.

  A. 늘려야하는 채널 만큼 zero padding

  B. 1x1 conv layer로 필요한 채널 만큼 convoultion 연산

Input의 size는 A, B 방법 모두 Stride을 2로 주어서 감소시켯다고 한다.

<a href='https://drive.google.com/drive/folders/1nRQo_fNmQMg7FkWrbZVUR87UOmk_AKJw?usp=sharing'>직접 구현한 ResNet34, ResNet50 모델의 flow chart</a>
