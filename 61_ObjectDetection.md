
![](http://i.imgur.com/y51J97v.png)


|년도|알고리즘|링크|입력|출력|특징|
|-|-|-|-|-|-|
|2014|R-CNN|[논문](https://arxiv.org/abs/1311.2524)|Image|Bounding boxes + labels for each object in the image.|AlexNet, 'Selective Search'사용 |
|2015|Fast R-CNN|[논문](https://arxiv.org/abs/1504.08083)|Images with region proposals.|Object classifications |Speeding up and Simplifying R-CNN, RoI Pooling|
|2016|Faster R-CNN|[논문](https://arxiv.org/abs/1506.01497),[한글](https://curt-park.github.io/2017-03-17/faster-rcnn/)| CNN Feature Map.|A bounding box per anchor|MS, Region Proposal|
|2017|Mask R-CNN|[논문](https://arxiv.org/abs/1703.06870)|CNN Feature Map.|Matrix with 1s on all locations|Facebook, pixel level|

![](http://i.imgur.com/qTRVI2j.png)







# paper

- [A Review on Deep Learning Techniques Applied to Semantic Segmentation ](https://arxiv.org/pdf/1704.06857v1.pdf)

- A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection (CVPR 2017): [논문](https://arxiv.org/abs/1704.03414), [GitHub](https://github.com/xiaolonw/adversarial-frcnn), [소개글](https://www.facebook.com/groups/TensorFlowKR/permalink/455680941439636/)


- [논문리뷰](http://blog.naver.com/PostList.nhn?blogId=kangdonghyun&from=postList&categoryNo=11#) : Mask R-CNN, YOLO

- [논문리뷰_글로벌 한량](http://man-about-town.tistory.com/category/-software/AI?page=1): Fast R-CNN ,SPP-Net, RCNN

- [논문리뷰_Jeongchul](http://jeongchul.tistory.com/category/MachineLearning): SSD, YOLO

# List

- list of [Mask R-CNN](http://forums.fast.ai/t/implementing-mask-r-cnn/2234)

- [awesome-object-proposals](https://github.com/caocuong0306/awesome-object-proposals)

# post

- [deep-object-detection-models](https://github.com/ildoonet/deep-object-detection-models): Deep Learning으로 학습된 Object Detection Model 에 대해 정리한 Archive 임.

## Image Segmentation

- [Fully Convolutional Networks for Semantic Segmentation](https://github.com/shelhamer/fcn.berkeleyvision.org): [정리](https://www.facebook.com/groups/AIKoreaOpen/permalink/1546985648668873/)


- [Image Segmentation](https://experiencor.github.io/segmentation.html)


- [Semantic Segmentation using Fully Convolutional Networks over the years](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html)

- [Image Segmentation using deconvolution layer in Tensorflow ](http://www.datasciencecentral.com/profiles/blogs/image-segmentation-using-deconvolution-layer-in-tensorflow)

## Object Recognition
- [A Brief History of CNNs in Image Segmentation: From R-CNN to Mask R-CNN](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4)

- ~~[추천: The Modern History of Object Recognition — 인포프래픽포함 ](https://medium.com/@nikasa1889/the-modern-history-of-object-recognition-infographic-aea18517c318)~~


- [Counting Objects with Faster R-CNN](https://softwaremill.com/counting-objects-with-faster-rcnn/)

- Object detection with neural networks — a simple tutorial using keras : [Blog](https://medium.com/towards-data-science/object-detection-with-neural-networks-a4e2c46b4491): [Youtube](https://www.youtube.com/watch?v=K9a6mGNmhbc&feature=youtu.be), [Code](https://github.com/jrieke/shape-detection)


- [Object Detection using Deep Learning for advanced users (Part-1)](https://medium.com/ilenze-com/object-detection-using-deep-learning-for-advanced-users-part-1-183bbbb08b19)


[A Brief History of CNNs in Image Segmentation: From R-CNN to Mask R-CNN](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4)



[How RPN (Region Proposal Networks) Works](https://www.youtube.com/watch?v=X3IlbjQs190): Youtube


[Counting Objects with Faster R-CNN](https://softwaremill.com/counting-objects-with-faster-rcnn/)



[QnA] [What are bounding box regressors doing in Fast-RCNN?](https://www.quora.com/Convolutional-Neural-Networks-What-are-bounding-box-regressors-doing-in-Fast-RCNN)


# Youtube


- [Computer Vision - StAR Lecture Series: Object Recognition](https://www.youtube.com/watch?v=fbFYdzatOMg) : MS




# Material 

- [Recent Progress on Object Detection_20170331](https://www.slideshare.net/JihongKang/recent-progress-on-object-detection20170331): ppt

- [ppt: Recent Progress on Object Detection_20170331](https://www.slideshare.net/JihongKang/recent-progress-on-object-detection20170331)


- [Lecture 6: CNNs for Detection,Tracking, and Segmentation](http://cvlab.postech.ac.kr/~bhhan/class/cse703r_2016s/csed703r_lecture6.pdf): 포항공대

-[Object Detection](http://slazebni.cs.illinois.edu/spring17/lec07_detection.pdf)

-[Single Shot MultiBox Detector와 Recurrent Instance Segmentation](https://www.slideshare.net/ssuser06e0c5/single-shot-multibox-detector-recurrent-instance-segmentation): 한글

# Implementation 

## YOLO
- YOLO9000: Better, Faster, Stronger : 모두의 연구소, [논문](https://arxiv.org/abs/1612.08242), [정리](http://www.modulabs.co.kr/DeepLAB_library/12796), [요약](https://www.facebook.com/groups/modulabs/permalink/1284949844903529/)

- [Basic Yolo with Keras](https://experiencor.github.io/yolo_keras.html)

- [YOLO](https://pjreddie.com/darknet/yolo/) : 실시간 Object탐지용


- [YOLO ver2 실습하기](http://blog.daum.net/sotongman/12):, [YOLO](http://blog.daum.net/sotongman/10)

## SSD

- [SSD Tensorflow](https://medium.com/@mslavescu/dhruv-parthasarathy-you-can-try-ssd-tensorflow-very-easily-especially-if-you-use-my-gtarobotics-1e515e693d51)

- [I created to test SSD on driving videos](https://github.com/OSSDC/SSD-Tensorflow/blob/master/notebooks/ossdc-vbacc-ssd_notebook.ipynb)

- [SSD: Single Shot MultiBox Detector in TensorFlow](https://github.com/OSSDC/SSD-Tensorflow)

- Keras SSD : [Keras v1](https://github.com/rykov8/ssd_keras), [Keras v2](https://github.com/cory8249/ssd_keras)



## u-net
* [U-net](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) : Biomedical Image Segmentation 참고용


## Full-Resolution Residual Networks

- Full-Resolution Residual Networks (FRRNs) for Semantic Image Segmentation in Street Scenes: [논문](https://arxiv.org/abs/1611.08323),[Code](https://github.com/TobyPDE/FRRN) [youtube](https://www.youtube.com/watch?v=PNzQ4PNZSzc&feature=youtu.be)

## Mask RCNN

- [Mask RCNN in TensorFlow](https://github.com/CharlesShang/FastMaskRCNN)

## Faster-RCNN

- [Tensorflow](https://github.com/smallcorgi/Faster-RCNN_TF)

- [Keras](https://github.com/yhenon/keras-frcnn)

- [Faster-R-CNN Install on Ubuntu 16.04(GTX1080 CUDA 8.0,cuDNN 5.1)](http://goodtogreate.tistory.com/entry/FasterRCNN-Install-on-Ubuntu-1604GTX1080-CUDA-80cuDNN-51): caffe, 한글,Good to Great 블로그

- [Faster R CNN Training](http://goodtogreate.tistory.com/558): caffe. 한글, Good to Great 블로그

- [Faster R-CNN](http://blog.daum.net/sotongman/9)

## Fast R-CNN

- [Fast R-CNN](http://blog.daum.net/sotongman/8)


## R-CNN 

- [keras-rcnn](https://github.com/broadinstitute/keras-rcnn)



- [R-CNN](http://blog.daum.net/sotongman/6)

## TensorBox

- [GitHub](https://github.com/TensorBox/TensorBox)

## Tensorflow Object Detection API

출처: http://goodtogreate.tistory.com/entry/Tensorflow-Object-Detection-API-SSD-FasterRCNN [GOOD to GREAT]

Single Shot Multibox Detector (SSD) with MobileNet
SSD with Inception V2
Region-Based Fully Convolutional Networks (R-FCN) with ResNet 101
Faster R-CNN with Resnet 101
Faster RCNN with Inception Resnet v2



[SPPnet](http://blog.daum.net/sotongman/7)
