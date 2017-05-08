


- [10 steps to bootstrap your machine learning project (part 1)](https://blog.metaflow.fr/10-steps-to-bootstrap-your-machine-learning-project-part-1-aa7e1031f5b1)

- [10 steps to bootstrap your machine learning project (part 2)](https://blog.metaflow.fr/10-steps-to-bootstrap-your-machine-learning-project-part-2-b6be78444c70)

- [Must Know Tips/Tricks in Deep Neural Networks (by  Vincent Granville)](http://www.datasciencecentral.com/profiles/blogs/must-know-tips-tricks-in-deep-neural-networks)

- [Must Know Tips/Tricks in Deep Neural Networks (by Xiu-Shen Wei)](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html)

- [Neural Network Tuning with TensorFlow](https://medium.com/computer-car/neural-network-tuning-with-tensorflow-cc14a23f132c)

- [How to Prevent Overfitting](http://sinahab.com/blog/how-to-prevent-overfitting/)

- [Model evaluation, selection and algorithm selection](https://tensorflow.blog/2017/03/30/model-evaluation-selection-and-algorithm-selection/)

Am I overfitting?
- Very high accuracy on the training dataset (eg: 0.99)
- Poor accuracy on the test data set (0.85)

- [머신러닝 모델 개발 삽질 경험기](http://bcho.tistory.com/1174): 조대협

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

# Limitation / Further Readings

- [catastrophic forgetting](https://deepmind.com/blog/enabling-continual-learning-in-neural-networks/)
