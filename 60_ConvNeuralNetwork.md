- [List of Awesome Deep Vision](https://github.com/kjw0612/awesome-deep-vision): Jiwon Kim

- [What is the class of this image ?](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html): 경진대회 우수 방식 및 논문 리스트

* [UCLA CNN] A Beginner’s Guide To Understanding Convolutional Neural Networks
  - [[Step1]](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks/?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
  - [[Step2]](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)
  - [[Step3]](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html?ref=mybridge.co?utm_source=mybridge&utm_medium=email&utm_campaign=read_more)

- [IMPLEMENTING A CNN FOR TEXT CLASSIFICATION IN TENSORFLOW](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

* 조대협의 초보자를 위한 컨볼루셔널 네트워크를 이용한 이미지 인식의 이해
  - [개요](http://bcho.tistory.com/1149)
  - [1/2](http://bcho.tistory.com/1156)
  - [예측](http://bcho.tistory.com/1157)
  - [실습](http://bcho.tistory.com/1154)
  - [영문1/2](http://www.kdnuggets.com/2016/09/beginners-guide-understanding-convolutional-neural-networks-part-1.html)
  - [영문2/2](http://www.kdnuggets.com/2016/09/beginners-guide-understanding-convolutional-neural-networks-part-2.html)

* [Deep Learning for Computer Vision – Introduction to Convolution Neural Networks](https://www.analyticsvidhya.com/blog/2016/04/deep-learning-computer-vision-introduction-convolution-neural-networks/)
* [The Neural Network Zoo](http://www.asimovinstitute.org/neural-network-zoo/?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)
* [Youtube:How an algorithm behind Deep Learning works](http://blog.revolutionanalytics.com/2016/09/how-the-algorithm-behind-deep-learning-works.html)[[정리]](http://www.kdnuggets.com/2016/08/brohrer-convolutional-neural-networks-explanation.html)
* [Understanding regularization for image classification and machine learning](http://www.pyimagesearch.com/2016/09/19/understanding-regularization-for-image-classification-and-machine-learning/)
* [딥 러닝을 위한 콘볼루션 계산 가이드](https://tensorflow.blog/a-guide-to-convolution-arithmetic-for-deep-learning/)
* [PPT:최성준 딥러닝 6주 수업](https://github.com/sjchoi86/dl-workshop/tree/master/presentations) : AlexNet, VGG net, GoogleLeNet, ResNet을 설명하고, Logistic regression, multilayer perceptron, convolutional neural network를 구현
* [ppt_Convolutional neural network in practice](http://www.slideshare.net/ssuser77ee21/convolutional-neural-network-in-practice)
* [Convolution arithmetic](https://github.com/vdumoulin/conv_arithmetic)
* [Deep Visualization Tool](http://yosinski.com/deepvis)
* [Image Completion with Deep Learning in TensorFlow](http://bamos.github.io/2016/08/09/deep-completion/?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more) : Brandon Amos, Ph.D at Carnegie Mellon University
* [7 Steps to Understanding Computer Vision](http://www.kdnuggets.com/2016/08/seven-steps-understanding-computer-vision.html?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+kdnuggets-data-mining-analytics+%28KDnuggets%3A+Data+Mining+and+Analytics%29)
* [Competition Scripts: Techniques for Tackling Image Processing](http://blog.kaggle.com/2016/06/17/competition-scripts-techniques-for-tackling-image-processing/)
* [Deep Learning Research Review: Generative Adversarial Nets](http://www.kdnuggets.com/2016/10/deep-learning-research-review-generative-adversarial-networks.html)
* [Deep Learning for Computer Vision – Introduction to Convolution Neural Networks](https://www.analyticsvidhya.com/blog/2016/04/deep-learning-computer-vision-introduction-convolution-neural-networks/)

* [이미지 분석에 꼭 필요한 딥러닝 기술을 정리한 자료](http://fbsight.com/t/topic/3024)

- [추천: Data Augmentation](https://www.facebook.com/groups/TensorFlowKR/permalink/436783573329373/): TensorFlowKR 글

* [ppt : Deep Learning in Computer Vision](https://www.slideshare.net/samchoi7/deep-learning-in-computer-vision-68541160) : Sungjoon Samuel 작성
* [ppt: Convolutional neural network in practice](https://www.slideshare.net/ssuser77ee21/convolutional-neural-network-in-practice)

* [ppt_CNN 초보자가 만드는 초보자 가이드 (VGG 약간 포함)](https://www.slideshare.net/leeseungeun/cnn-vgg-72164295)

# Paper
- [Large-Scale Evolution of Image Classifiers](https://arxiv.org/abs/1703.01041)

- [Faster R-CNN한글 정리](https://curt-park.github.io/2017-03-17/faster-rcnn/)

- [Mask R-CNN](https://arxiv.org/abs/1703.06870): 얀쿤 추천 논문
```
오늘 소개해 드릴 논문은 A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection입니다. CVPR 2017에 나올 논문으로 CMU의 The Robotics Institute에서 나온 논문입니다.
논문에서는 object detection system에서의 방법이라고 했지만, 요즘 딥러닝 영상 처리 테스크의 발전을 위해 크게는 3가지 방식이 주로 시도됩니다. (다른 방법도 많지만..)
1. 새로운 deeper network를 만들려는 연구.
2. contextual reasoning을 사용하는 연구.
3. Data 자체를 더 잘 사용하려는 연구.
이 논문은 3번에 관한 겁니다. 3번의 쉬운 예제로는 data augmentation, noise modeling, adversarial example이나 generated example을 이용한 학습등이 있습니다. 이런 방법들은 주로 데이터의 부재 또는 부족 때문에 많이 시도되었습니다. 그러나 아래 그림에서 볼 수 있듯이 저희가 부족하다고 생각하는 occlusion이나 deformation 데이터는 다른 데이터에 비해 적습니다. 그런데, 이런 데이터를 GAN 등의 기술로 재현을 잘 한다할 수 있다고손 치더라도, 그러기 위해서는 많은 데이터가 필요합니다. 그러니 부족한 데이터를 극복하기 위해 generated data를 생성하기 위해서는 많은 데이터가 필요하다!! 라는 무한 루프 속으로 가죠. 그래서 이 논문은 이렇게 occlusion과 deformation을 포함한 "hard"한 예제를 데이터도 많이 필요하고, 어려운 픽셀레벨에서의 생성 방법이 아닌 다른 방법으로 생성할 수 없을까?란 질문에서 출발합니다.
이 논문에서의 그 질문의 답은 pixel 레벨에서 하기 힘들다면 feature map에서 하면 어떨까? 란 방향을 제시합니다. 판단을 힘들게 하는 blocking mask를 학습하는 네트웍을 사용하여 하나의 예제를 occlusion이 있는거 처럼 어렵게 만드는 것이죠. 그렇게 어려워진 예제를 가지고 학습을 하는 방향입니다. 위와 같이 occlusion을 조금 해결했다면.. deformation은? 이란 질문에 ... 우리가 그걸 위해 좀 쓰던 data augmentation이 있죠. rotation ... 그걸 위해 위와 같은 걸 하나 더 답니다. 아.. rotation하면 생각나는 네트웍이 있으시죠? Spatial Transformer Network (STN)...그걸 더 답니다. 이것도 위와 같이 학습.. 하면 판단을 어렵게 하게 하는 거죠.
결과를 보면 전체적으로 기존 방법보다 좋아지는 걸 관찰할 수 있습니다.
이 논문을 소개시켜드린 것은 오늘 김홍배 박사님이 data augmentation에 대해 문의하셨는데.. 위와 같은 방식으로 하면 일종의 hard example을 생성하는 data augmentation을 학습하는 방법으로 생각하실 수 있습니다. 최근 Smart Augmentation Learning an Optimal Data Augmentation Strategy (https://arxiv.org/pdf/1703.08383.pdf)과 같은 논문에서는 학습 성능을 향상시키는 data augmentation을 학습시키려고 했습니다. 이번 논문을 다시 생각하면 occlusion이나 deformation을 하나의 data augmentation이라고 생각하면 hard example을 만드는 data augmentation을 학습하는 구조로 이해하실 수도 있을거 같습니다.

```

- A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection (CVPR 2017): [논문](https://arxiv.org/abs/1704.03414), [GitHub](https://github.com/xiaolonw/adversarial-frcnn)
