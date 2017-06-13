
![](http://i.imgur.com/y51J97v.png)


|년도|알고리즘|링크|입력|출력|특징|
|-|-|-|-|-|-|
|2014|R-CNN|[논문](https://arxiv.org/abs/1311.2524)|Image|Bounding boxes + labels for each object in the image.|AlexNet, 'Selective Search'사용 |
|2015|Fast R-CNN|[논문](https://arxiv.org/abs/1504.08083)|Images with region proposals.|Object classifications |Speeding up and Simplifying R-CNN, RoI Pooling|
|2016|Faster R-CNN|[논문](https://arxiv.org/abs/1506.01497),[한글](https://curt-park.github.io/2017-03-17/faster-rcnn/)| CNN Feature Map.|A bounding box per anchor|MS, Region Proposal|
|2017|Mask R-CNN|[논문](https://arxiv.org/abs/1703.06870)|CNN Feature Map.|Matrix with 1s on all locations|Facebook, pixel level|

> 출처 : [A Brief History of CNNs in Image Segmentation: From R-CNN to Mask R-CNN](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4)

# paper

- [A Review on Deep Learning Techniques Applied to Semantic Segmentation ](https://arxiv.org/pdf/1704.06857v1.pdf)

- A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection (CVPR 2017): [논문](https://arxiv.org/abs/1704.03414), [GitHub](https://github.com/xiaolonw/adversarial-frcnn)



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





# List

- list of [Mask R-CNN](http://forums.fast.ai/t/implementing-mask-r-cnn/2234)

# post

- [추천: The Modern History of Object Recognition — 인포프래픽포함 ](https://medium.com/@nikasa1889/the-modern-history-of-object-recognition-infographic-aea18517c318)

- [ppt: Recent Progress on Object Detection_20170331](https://www.slideshare.net/JihongKang/recent-progress-on-object-detection20170331)

- [Image Segmentation](https://experiencor.github.io/segmentation.html)

- [Fully Convolutional Networks for Semantic Segmentation](https://github.com/shelhamer/fcn.berkeleyvision.org): [정리](https://www.facebook.com/groups/AIKoreaOpen/permalink/1546985648668873/)

- <del>[Counting Objects with Faster R-CNN](https://softwaremill.com/counting-objects-with-faster-rcnn/)</del>

- Object detection with neural networks — a simple tutorial using keras : [Blog](https://medium.com/towards-data-science/object-detection-with-neural-networks-a4e2c46b4491): [Youtube](https://www.youtube.com/watch?v=K9a6mGNmhbc&feature=youtu.be), [Code](https://github.com/jrieke/shape-detection)


# YOLO
- YOLO9000: Better, Faster, Stronger : 모두의 연구소, [논문](https://arxiv.org/abs/1612.08242), [정리](http://www.modulabs.co.kr/DeepLAB_library/12796), [요약](https://www.facebook.com/groups/modulabs/permalink/1284949844903529/)

- [Basic Yolo with Keras](https://experiencor.github.io/yolo_keras.html)

- [YOLO](https://pjreddie.com/darknet/yolo/) : 실시간 Object탐지용


# SSD

- [SSD Tensorflow](https://medium.com/@mslavescu/dhruv-parthasarathy-you-can-try-ssd-tensorflow-very-easily-especially-if-you-use-my-gtarobotics-1e515e693d51)

- [I created to test SSD on driving videos](https://github.com/OSSDC/SSD-Tensorflow/blob/master/notebooks/ossdc-vbacc-ssd_notebook.ipynb)

- [SSD: Single Shot MultiBox Detector in TensorFlow](https://github.com/OSSDC/SSD-Tensorflow)

- Keras SSD : [Keras v1](https://github.com/rykov8/ssd_keras), [Keras v2](https://github.com/cory8249/ssd_keras)

# u-net
* [U-net](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) : Biomedical Image Segmentation 참고용


# Full-Resolution Residual Networks

- Full-Resolution Residual Networks (FRRNs) for Semantic Image Segmentation in Street Scenes: [논문](https://arxiv.org/abs/1611.08323),[Code](https://github.com/TobyPDE/FRRN) [youtube](https://www.youtube.com/watch?v=PNzQ4PNZSzc&feature=youtu.be)

# Faster-RCNN

- [Tensorflow](https://github.com/smallcorgi/Faster-RCNN_TF)

- [Keras](https://github.com/yhenon/keras-frcnn)

# Mask RCNN

- [Mask RCNN in TensorFlow](https://github.com/CharlesShang/FastMaskRCNN)

# R-CNN for keras

- [keras-rcnn](https://github.com/broadinstitute/keras-rcnn)

# TensorBox

- [GitHub](https://github.com/TensorBox/TensorBox)
