# List
- [Bayesian Deep Learning](http://bayesiandeeplearning.org): NIPS 2016 Workshop


# Article/posts

- [Deep Learning Is Not Good Enough, We Need Bayesian Deep Learning for Safe AI](http://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/)

1. What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? [[논문]](https://arxiv.org/pdf/1703.04977.pdf)
2. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. [[논문]](https://arxiv.org/pdf/1705.07115.pdf)
3. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning [[논문]](http://proceedings.mlr.press/v48/gal16.pdf)

오늘은 논문 소개라기 보다는 논문을 추천 드리고자 합니다. 만약 하시는 분야가 자율 주행, 의료와 같이 딥러닝 판단의 실패가 크리티컬한 분야에 계시는 분은 다 공감하실 글과 관련 논문입니다. 작년 Dropout as a Bayesian Approximation 논문으로 딥러닝에서의 uncertainty에 관련해서 이슈를 불러온 Yarin Gal 팀이 낸 논문 두편입니다.
블로그 글의 주인공과 Yarin Gal은 What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision을 통해 작년 논문에서 언급한 uncertainty를 구분하고, 그 것을 모델링하는 방법에 대해 논하고 있습니다. 모델이 뭔가를 헷갈릴 때, 왜 헷갈리게 되는지에 따라 크게 두개로 분류하고 나눴습니다. 아예 보지 못 했던 데이터에 대해서 헷갈리는 경우, 특정 도메인에서 발생되는 노이즈에 의해 발생되는 경우 등 여러 분류를 해 더 잘 분석할 수 있고, 그 것을 모델링할 수 있는 방안에 대해 논하였고, 두번째 논문인 Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics에서 그것을 참신하게 사용하는 예제를 보여줍니다.
전 케글이나 이미지넷에 나오는 메트릭이 저희가 접하는 모든 분야에 대해 대변하지 못 하고, 그렇기에 이런 불확실성에 대한 메트릭이 중요한 분야가 있다고 생각합니다. 단순히 벤치마킹 대회에서 1위를 목표로 한다면 모르지만, 고객을 상대해야 하는 서비스를 위해 이 분야 기술을 하신다면 새로나온 두개의 논문을 추천드립니다. (어차피 1,2번 논문을 읽다보면 3번 논문에 손이 가게되는 구조라.. )

- [Everything that Works Works Because it's Bayesian: Why Deep Nets Generalize?](http://www.inference.vc/everything-that-works-works-because-its-bayesian-2/)

- [한글: 베이지안 딥러닝](https://drive.google.com/file/d/0B8v4MKOWQrJAa3J1bXRFTERWbEU/view): 서울대학교, 김용대교수

# QnA

- [What is exactly a Bayesian guy in machine learning?](https://www.reddit.com/r/MachineLearning/comments/6dbwnf/d_what_is_exactly_a_bayesian_guy_in_machine/): Raddit
