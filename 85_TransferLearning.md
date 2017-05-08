![](https://pbs.twimg.com/media/C7fbqxLW4AERcdD.jpg)

[Tensorflow 공식 Inception GitHUb](https://github.com/tensorflow/models/tree/master/inception)

# Article / post
- <del>[추천] CS231n : Transfer Learning and Fine-tuning Convolutional Neural Networks: [원문](http://cs231n.github.io/transfer-learning), [번역](http://ishuca.tistory.com/entry/CS231n-Transfer-Learning-and-Finetuning-Convolutional-Neural-Networks-한국어-번역)</del>

- <Del>[추천] [Transfer Learning - Machine Learning's Next Frontier](http://sebastianruder.com/transfer-learning/index.html)</del> 추천

- <del>[Using Transfer Learning and Bottlenecking to Capitalize on State of the Art DNNs](https://medium.com/@galen.ballew/transferlearning-b65772083b47), [Transfer Learning in Keras](https://galenballew.github.io//articles/transfer-learning/): 후반 텐서보드 코드  [GitHub](https://github.com/galenballew/transfer-learning)</del>

- <del>[TensorFlow 와 Inception-v3 를 이용하여 원하는 이미지 학습과 추론 해보기](http://gusrb.tistory.com/m/16): retrain.py코드를 이용한 실행, 코드 자체에 대한 이해필요</del>, [Google CodeLAb 실습](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html?index=..%2F..%2Findex#0)

- <del>[텐서플로우(TensorFlow)를 이용한 ImageNet 이미지 인식(추론) 프로그램 만들기](http://solarisailab.com/archives/346?ckattempt=1): solarisailab작성, 코드 포함</del>, 간단하나 TF코드 이해 필요

- <del>[CNNs in Practice](http://nmhkahn.github.io/CNN-Practice): 중간에 transfer-learning에 대한여 잠깐 언급 </del>

- <del>[DeepMind just published a mind blowing paper: PathNet.](http://www.kcft.or.kr/2017/02/2120): 미래금융연구센터 한글 정리 자료 </del>

- <del>[초짜 대학원생의 입장에서 이해하는 Domain-Adversarial Training of Neural Networks (DANN)](http://jaejunyoo.blogspot.com/2017/01/domain-adversarial-training-of-neural.html): 중간에 transfer-learning에 대한여 잠깐 언급</del>

- [PPT: Transfer Learning and Fine Tuning for Cross Domain Image Classification with Keras](https://www.slideshare.net/sujitpal/transfer-learning-and-fine-tuning-for-cross-domain-image-classification-with-keras)

- <del>[Transfer Learning using Keras](https://medium.com/towards-data-science/transfer-learning-using-keras-d804b2e04ef8)</del>

- [ImageNet: VGGNet, ResNet, Inception, and Xception with Keras](http://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)

- Transfer learning using pytorch: Vishnu Subramanian, [Part 1](https://medium.com/towards-data-science/transfer-learning-using-pytorch-4c3475f4495), [Part 2](https://medium.com/towards-data-science/transfer-learning-using-pytorch-part-2-9c5b18e15551)

- [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html): Keras 공식 자료, [[GitHub]](https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069), [[VGG16_weight]](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)

- [Globa Average Pooling](https://www.quora.com/What-is-global-average-pooling)을 사용함으로써 기존 학습시 사용했던 학습의 제약 없이 아무 Tensor/image size를 사용할수 있음

> What is important to know about global average pooling is that it allows the network to accept any Tensor/image size, instead of expecting the size that it was originally trained on.


# paper
- [A Survey on Transfer Learning ](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf):SJ Pan et al. 2010 [ppt](https://www.slideshare.net/azuring/a-survey-on-transfer-learning)

- [A survey of transfer learning](https://goo.gl/e87S9y): Karl Weiss, 2016, 40P

- [Transfer Schemes for Deep Learning in Image Classification, 2015,87P ](http://webia.lip6.fr/~carvalho/download/msc_micael_carvalho_2015.pdf):

- [How transferable are features in deep neural networks?,2014](https://arxiv.org/abs/1411.1792)

- [CNN Features off-the-shelf: an Astounding Baseline for Recognition,2014](https://arxiv.org/abs/1403.6382)

# Implementation

- [Udacity Deeplearning - Transfer Learning](https://github.com/udacity/deep-learning/tree/master/transfer-learning):

- [Convert Caffe models to TensorFlow.](https://github.com/ethereon/caffe-tensorflow)

- [Inception in TensorFlow](https://github.com/tensorflow/models/tree/master/inception): 텐서플로우 공식 코드

- [Jupyter: 08_Transfer_Learning](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/08_Transfer_Learning.ipynb): Hvass, 동영상

- [keras-transfer-learning](https://github.com/neocortex/keras-transfer-learning): 추천, 5개의 노트북으로 구성, TL은 3,4번 노트북

- [deep-photo-styletransfer](https://github.com/luanfujun/deep-photo-styletransfer)

- [Keras Exampls](https://github.com/fchollet/keras/tree/master/examples): fchollet

- [Fine-Tune CNN Model via Transfer Learning](https://github.com/abnera/image-classifier): abnera, [Kaggle](https://www.kaggle.com/abnera/dogs-vs-cats-redux-kernels-edition/transfer-learning-keras-xception-cnn)

* [jupyter_nception v3](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/07_Inception_Model.ipynb),[[논문]](http://arxiv.org/pdf/1512.00567v3.pdf),[[New Version]](https://research.googleblog.com/2016/08/improving-inception-and-image.html)

- [Transfer Learning and Fine Tuning for Cross Domain Image Classification with Keras](https://github.com/sujitpal/fttl-with-keras): sujitpal

- [My experiments with AlexNet, using Keras and Theano](https://rahulduggal2608.wordpress.com/2017/04/02/alexnet-in-keras/): [GithubUb](https://github.com/duggalrahul/AlexNet-Experiments-Keras), [reddit](https://www.reddit.com/r/MachineLearning/comments/64cs0a/p_releasing_codes_for_training_alexnet_using_keras/?st=j1d0iu2p&sh=4bf414d8)
  - AlexNet operates on 227×227 images.

- [Keras pretrained models (VGG16 and InceptionV3) + Transfer Learning for predicting classes in the Oxford 102 flower dataset](https://github.com/Arsey/keras-transfer-learning-for-oxford102):


- [Heuritech project](https://github.com/heuritech/convnets-keras)

- [CarND-Transfer-Learning-Lab](https://github.com/paramaggarwal/CarND-Transfer-Learning-Lab)

- 매틀랩: [Transfer Learning Using Convolutional Neural Networks](https://www.mathworks.com/help/nnet/examples/transfer-learning-using-convolutional-neural-networks.html)

- 매틀랩: [Transfer Learning and Fine-Tuning of Convolutional Neural Networks](http://www.mathworks.com/help/nnet/examples/transfer-learning-and-fine-tuning-of-convolutional-neural-networks.html)
