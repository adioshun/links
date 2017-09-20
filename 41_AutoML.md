

- [The Current State of Automated Machine Learning](http://www.kdnuggets.com/2017/01/current-state-automated-machine-learning.html)

# Paper

- [Neural Architecture Search with Reinforcement Learning](https://openreview.net/pdf?id=r1Ue8Hcxg): [정리_한국어](https://www.slideshare.net/KihoSuh/neural-architecture-search-with-reinforcement-learning-76883153)
>  Google AutoML : 머신러닝 개발 업무 중 일부를 자동화하는 프로젝트. --> 딥러닝을 이용해서 딥러닝을 만든다면? 딥러닝하는 개발자들의 설자리는 어떻게 될까?
- Network Architecture의 hyper parameter들을 찾는 방법이 딱히 없는 것이 현실이다.
- 좋은 Neural Network Architecture 를 찾는 Neural Network를 찾아보자!
- Configuration String : filter width/filter height/stride/#of filter 등등
- RNN을 이용해서 configuration string 을 만들어줌. (그림 1 참고)
- 이 RNN에서 나온 configuration을 가지고 NN을 만들어주는데(이게 우리가 설계한 nework). 이것을 child network라고 부름.
- child network의 accuracy 에 따라 RNN(policy network)의reward를 준다. (RNN은 강화학습방법 중 하나인 REINFORCE로 학습을 시킴)
- (질문 1) 그렇다면 이 네트워크를 설계하는 RNN 의 입력으로는 무엇을 주는 것일까?
답은 여러분의 마음속에 ^^;; (안줘도 되는 것 같다는게 저의 추측....)
- REINFORCE(monte-carlo계열)에서 항상 발생하는 high variance 문제를 없애주기 위해, baseline을 제거
(R 대신 R-b를 사용)
- CIFAR10 문제를 해결하기 위해 GPU 800개를 2-3주 돌려서 구했다고 함. 헐.. ㄷㄷㄷ
- training 속도를 빠르게 하기 위해, distributed training 기법을 사용 (그림 2 참고)
- search space를 widening 하기 위해, skip connection을 사용할 수도 있다. (그림 3 참고.) (음.. 이건 search space가 왜 widening되는지는 살짝 이해가 안감)
- 그래서 이 논문으로 구한 최종 network 구조는 -> 그림 4참고.
<특징 1> 첫번째 레이어 출력이 다른 레이어로 엄청나게 많은 skip connection이 연결되어 있다.
<특징 2> 직사각형 필터가 의외로 많이 사용됨.
- CNN 사이즈 얼마나 할지.. 이런거 말고, LSTM 같은 구조를 새롭게 만들수도 있다. 그렇게해서 만들어봤고, GNMT 의 LSTM을 대체했더니 성능도 올라가더라.

> 머신러닝 개발 업무를 자동화하는 구글의 AutoML이 뭘하려는지 이 논문을 통해 잘 보여줍니다.
이 논문에서는 뉴럴 네트워크 구조를 만드는 뉴럴 네트워크 구조에 대해서 설명합니다. 800개의 GPU를 혹은 400개의 CPU를 썼고 State of Art 혹은 State of Art 바로 아래이지만 더 빠르고 더 작은 네트워크를 이것을 통해 만들었습니다. 이제 Feature Engineering에서 Neural Network Engineering으로 페러다임이 변했는데 이것의 첫 시도 한 논문입니다.

# Post / Article

- [Automating automation: framework for developing deep learning models](http://www.techleer.com/articles/188-automating-automation-framework-for-developing-deep-learning-models/)

- [Overview of Hyperparameter Tuning](https://cloud.google.com/ml-engine/docs/concepts/hyperparameter-tuning-overview):

- [Design by evolution](https://medium.com/@stathis/design-by-evolution-393e41863f98):  딥네트워크 아키텍처와 하이퍼파라미터 등을 설정하는데에 Evolutionary Computing, Genetic Algorithm 등이 다시 부각이 되고 있죠? PyTorch를 이용해서 이 내용들을 설명하면서 간단한 예제

- [Accelerating Deep Learning Research with the Tensor2Tensor Library](https://research.googleblog.com/2017/06/accelerating-deep-learning-research.html): open-source system for training deep learning models in TensorFlow.

- [Automate your Machine Learning in Python – TPOT and Genetic Algorithms](https://blog.alookanalytics.com/2017/05/25/automate-your-machine-learning/)


# Material (ppt, pdf)

- [automl & autodraw](https://www.slideshare.net/taeyounglee1447/io17ex-automl-autodraw): ppt

# Implementation 

- [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://blog.openai.com/evolution-strategies/): [Keras구현코드](https://gist.github.com/nicksam112/00e9638c0efad1adac878522cf172484), [Evostra: Evolution Strategy for Python](https://github.com/alirezamika/evostra): Evolutio Strategy (ES) is an optimization technique based on ideas of adaptation and evolution.
