# 스스로 모델 만들기

- [github:DEvol - Deep Neural Network Evolution](https://github.com/joeddav/devol)

DeepLearning Project Workflow라고 읽고
Bias-Variance Tradeoff에 가까운데 가끔 비슷한 질문이 올라오는 것 같아서 올립니다.

모델 성능이 원하는대로 나오지 않을때

1) 데이터를 더 모아야 하는지
2) 더 큰 모델을 만들어야 할 지

High bias, High Variance 체크를 통해 간단히 판단하는 법입니다.
http://fbsight.com/t/deeplearning-project-workflow-bias-variance-tradeoff/78035


# Paper

-


# Article / Post

- [Applying deep learning to real-world problems](https://medium.com/merantix/applying-deep-learning-to-real-world-problems-ba2d86ac5837)
  - Learning I: the value of pre-training
  - Learning II: caveats of real-world label distributions
  - Learning III: understanding black box models

- <del>[14 DESIGN PATTERNS TO IMPROVE YOUR CONVOLUTIONAL NEURAL NETWORKS](http://www.topbots.com/14-design-patterns-improve-convolutional-neural-network-cnn-architecture/)>/del>: 추천

- 10 steps to bootstrap your machine learning project: [(part 1)](https://blog.metaflow.fr/10-steps-to-bootstrap-your-machine-learning-project-part-1-aa7e1031f5b1), [(part 2)](https://blog.metaflow.fr/10-steps-to-bootstrap-your-machine-learning-project-part-2-b6be78444c70)]

- Improve your neural networks:  [(Part 1)](http://adventuresinmachinelearning.com/improve-neural-networks-part-1/), [(Part 1한글)](https://www.nextobe.com/single-post/2017/05/11/Neural-Network-%25EA%25B0%259C%25EC%2584%25A0)

- [Must Know Tips/Tricks in Deep Neural Networks (by  Vincent Granville)](http://www.datasciencecentral.com/profiles/blogs/must-know-tips-tricks-in-deep-neural-networks)

- [Must Know Tips/Tricks in Deep Neural Networks (by Xiu-Shen Wei)](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html)

- [Neural Network Tuning with TensorFlow](https://medium.com/computer-car/neural-network-tuning-with-tensorflow-cc14a23f132c)

- [Machine Learning Pipeline](http://bcho.tistory.com/1177): 조대협, 머신러닝 시스템 프로세스와 아키텍쳐

- [How to Prevent Overfitting](http://sinahab.com/blog/how-to-prevent-overfitting/)

- [Model evaluation, selection and algorithm selection](https://tensorflow.blog/2017/03/30/model-evaluation-selection-and-algorithm-selection/)

Am I overfitting?
- Very high accuracy on the training dataset (eg: 0.99)
- Poor accuracy on the test data set (0.85)

- [머신러닝 모델 개발 삽질 경험기](http://bcho.tistory.com/1174): 조대협

- [학습 조기종료 시키기](https://tykimos.github.io/Keras/2017/07/09/Early_Stopping/): 조기 종료하는 시점

- [Pytorch 를 이용한 fine-tuning 및 feature extraction 코드](https://github.com/meliketoy/fine-tuning.pytorch)

# [추천]전처리
- [cifarnet_preprocessing.py](https://github.com/tensorflow/models/tree/master/slim/preprocessing)
- inception_preprocessing.py
- lenet_preprocessing.py
- preprocessing_factory.py
- vgg_preprocessing.py

- [Where do I call the BatchNormalization function in Keras?](http://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras)

- [How To Improve Deep Learning Performance](http://machinelearningmastery.com/improve-deep-learning-performance/)

- Catastrophic forgetting 결함에 대한 문제를 극복하는 방법: [논문](https://arxiv.org/pdf/1612.00796.pdf), [코드](https://github.com/ariseff/overcoming-catastrophic)

Some important parameters to look out for while optimizing neural networks are:
- Type of architecture
- Number of Layers
- Number of Neurons in a layer
- Regularization parameters
- Learning Rate
- Type of optimization / backpropagation technique to use
- Dropout rate
- Weight sharing
- etc.
- etc for CNN : convolutional filter size, pooling value, etc.
Here are some resources for tips and tricks for training neural networks. ([Resource 1](http://cs231n.github.io/neural-networks-3/#baby), [Resource 2](https://www.quora.com/Machine-Learning-What-are-some-tips-and-tricks-for-training-deep-neural-networks), [Resource 3](https://arxiv.org/abs/1206.5533))


# Tools

-[Xcessiv](https://github.com/reiinakano/xcessiv): 웹에서 모델과 데이터를 지정하고 하이퍼파라미터 탐색을 위한 조건을 걸어서 자동으로 튜닝해주는 코드

-

# 나만의 모델을 만들어 보자
* 참고 [[Youtube]](https://www.youtube.com/watch?v=076pp-42unI),  [[ppt]](http://www.slideshare.net/carpedm20/ss-63116251)



- [추천][Use Keras Deep Learning Models with Scikit-Learn in Python](http://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/): Keras와 Scikit-Learn을 이용한 성능향상

* [Evaluate the Performance Of Deep Learning Models in Kerasn](http://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/)

- [How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras](http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)





# QnA

- [Why does learning curve of my model shows large variance in training error? How to fix it?](https://www.reddit.com/r/MachineLearning/comments/65rnyj/dwhy_does_learning_curve_of_my_model_shows_large/?st=j1lduqn0&sh=a0d0d41f)

- [Live loss plots inside Jupyter Notebook for Keras](https://www.reddit.com/r/MachineLearning/comments/65jelb/d_live_loss_plots_inside_jupyter_notebook_for/?st=j1k2qa06&sh=01fc92df)

- [How big should batch size and number of epochs be when fitting a model in Keras?](http://stackoverflow.com/questions/35050753/how-big-should-batch-size-and-number-of-epochs-be-when-fitting-a-model-in-keras)

- [Hyperparameter search benchmark?](https://www.reddit.com/r/MachineLearning/comments/69n74f/p_hyperparameter_search_benchmark/)

- <del>[How to design deep convolutional neural networks?](http://stackoverflow.com/questions/37280910/how-to-design-deep-convolutional-neural-networks): </del> 결론은 정해진 룰 없음. 경험적으로 반복 수행으로 찾아 내는것임

# Limitation / Further Readings

- [catastrophic forgetting](https://deepmind.com/blog/enabling-continual-learning-in-neural-networks/)
