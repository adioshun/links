- [List of GAN Implementation](https://github.com/wiseodd/generative-models): wiseodd, 설명 [BLog](http://wiseodd.github.io/techblog/)


# Article / Blog

- 지적 대화를 위한 깊고 넓은 딥러닝 (Feat. TensorFlow): [[추천]: youtube](https://youtu.be/soJ-wDOSCf4?t=1m47s), 김태훈, PyCon APAC 2016

- [TensorFlowKR_PR12모임 GAN 설명](https://www.facebook.com/groups/TensorFlowKR/permalink/456848987989498/)


* [On the intuition behind deep learning & GANs — towards a fundamental understanding](https://medium.com/waya-ai/introduction-to-gans-a-boxing-match-b-w-neural-nets-b4e5319cc935#.e6alt2dpu)

* [Generative Adversarial Networks (GANs) in 50 lines of code (PyTorch)](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.pjokxgjca)

- [How do GANs intuitively work?](https://hackernoon.com/how-do-gans-intuitively-work-2dda07f247a1#.jmn4i02yi)

- [Fantastic GANs and where to find them](http://guimperarnau.com/blog/2017/03/Fantastic-GANs-and-where-to-find-them): GAN의 여러 모형들 설명

- <del>[GAN 그리고 Unsupervised Learning](http://t-robotics.blogspot.com/2017/03/gan-unsupervised-learning.html#.WNGonCErJxB): 개요로 읽기 적당한글, t-robotics블로그, 테크M 투고글</del>

* InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets: [논문](https://arxiv.org/abs/1606.03657),  [ppt정리](http://www.slideshare.net/ssuser06e0c5/infogan-interpretable-representation-learning-by-information-maximizing-generative-adversarial-nets-72268213)

- [아주 간단한 GAN구현하기](http://blog.naver.com/atelierjpro/220984758512): 참고 [GitHub](https://github.com/hwalsuklee/tensorflow-GAN-1d-gaussian-ex), Generative Adversarial Network for approximating a 1D Gaussian distribution

* NIPS 2016 Tutorial:Generative Adversarial Networks: [논문](https://arxiv.org/pdf/1701.00160v1.pdf),[youtube](http://fbsight.com/t/goodfellow-gan-nips-2016-tutorial/59058), 논문 및 저자 설명

- [유투브 강의:How to Generate Images with Tensorflow (LIVE) ](https://www.youtube.com/watch?v=iz-TZOEKXzA&feature=youtu.be)

# 논문
* [Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828v1.pdf) : Ashish Shrivastava, Apple, 216
 * the blueprint for training state-of-the-art neural nets from only __synthetic__ and unlabelled data
 * 영문 정리 글 : [SimGANs - a game changer in unsupervised learning, self driving cars, and more](https://medium.com/waya-ai/simgans-applied-to-autonomous-driving-5a8c6676e36b#.tcbuo9za5)
- Improved Techniques for Training GANs: [논문](https://arxiv.org/abs/1606.03498)

### 초짜 대학원생 입장에서 이해하는

- Generative Adversarial Nets논문 분석 : [# 1](http://jaejunyoo.blogspot.com/2017/01/generative-adversarial-nets-1.html), [# 2](http://jaejunyoo.blogspot.com/2017/01/generative-adversarial-nets-2.html)

- Domain-Adversarial Training of Neural Networks (DANN) : [#1](http://jaejunyoo.blogspot.com/2017/01/domain-adversarial-training-of-neural.html), [#2](http://jaejunyoo.blogspot.com/2017/01/domain-adversarial-training-of-neural-2.html), [#3](http://jaejunyoo.blogspot.com/2017/01/domain-adversarial-training-of-neural-3.html)

- Deep Convolutional Generative Adversarial Network (DCGAN): [#1](http://jaejunyoo.blogspot.com/2017/02/deep-convolutional-gan-dcgan-1.html), [#2](http://jaejunyoo.blogspot.com/2017/02/deep-convolutional-gan-dcgan-2.html)

- Unrolled Generative Adversarial Networks: [#1](http://jaejunyoo.blogspot.com/2017/02/unrolled-generative-adversarial-network-1.html), [#2](http://jaejunyoo.blogspot.com/2017/02/unrolled-generative-adversarial-network-2.html)


# 구현
* [SimGan](https://github.com/wayaai/SimGAN): Keras 코드

- An introduction to Generative Adversarial Networks (with code in TensorFlow): [원문](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/), <del>[정리](http://keunwoochoi.blogspot.com/2016/12/generative-adversarial-network-gan.html)</del>, [GitHub](https://github.com/AYLIEN/gan-intro)

- Least Squares Generative Adversarial Networks[(LSGAN)](https://github.com/GunhoChoi/GAN_simple/blob/master/LSGAN/LSGAN_TF.ipynb), [구현시 어려웠던점](https://m.facebook.com/groups/TensorFlowKR/?view=permalink&id=434490703558660)

* [SimGAN_NYU_Hand](https://github.com/shinseung428/simGAN_NYU_Hand) : Simulated+Unsupervised (S+U) learning in TensorFlow /w NYU Hand Dataset

- [Generative Adversarial Networks (GANs) in 50 lines of code (PyTorch)](https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.6uamvl61e): Dev Nag's 미디엄 포스트

- [Keras Adversarial Models](https://github.com/bstriner/keras-adversarial)

- [DiscoGAN](https://github.com/SKTBrain/DiscoGAN): SKTBrain, Pythrch기반
  - [DiscoGAN 설명자료](https://www.facebook.com/notes/sk-t-brain/sk-t-brain-research/398821727155314)
  - [DiscoGAN Arxiv 논문 링크](https://arxiv.org/abs/1703.05192)
  - [Taehoon Kim 님이 구현하신 DiscoGAN 소스코드](https://github.com/carpedm20/DiscoGAN-pytorch)
  - [wiseodd 님이 구현하신 DiscoGAN 소스코드](https://github.com/…/generative-m…/tree/master/GAN/disco_gan)
- [List of generative models](https://github.com/wiseodd/generative-models): 거의 모든 GAN코드 모음

- [DCGAN-MNIST](https://github.com/erilyth/DCGANs): Keras버젼, erilyth

- [Variational Auto-Encoder (VAE)](https://github.com/hwalsuklee/tensorflow-mnist-VAE)


- Deep Convolutional GAN(DCGAN): Tensorflow,  [code](https://github.com/carpedm20/DCGAN-tensorflow),  [demo](http://carpedm20.github.io/faces/)
