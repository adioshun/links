

# Article/post

- ["추론도 가능하다"…AI 연구 새 이정표 썼다](http://m.zdnet.co.kr/news_view.asp?article_id=20170609151041#imadnews): zdnet뉴스

[A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf)
- 시각인식과 문장이해가 결합된 관계추론(RN : Relational Reasoning)에 관한 딥마인드의 논문
- 딥마인드

- 그렇지만, 여전히 해결하지 못했던 것이 복잡한 관계를 추론하고, 이에 대한 답을 내는 것에 있어서는 인간과 비교할 수 없는 수준이었습니다. 이 문제를 해결하기 위해서 그 동안 신경망을 그래프 구조의 노드가 아닌 관계를 설명하는 엣지에 더 중점을 둔 연구들이 최근들어 계속 나오고 있었습니다. 대표적인 것들이 Graph Neural Networks, Gated Graph Sequence Neural Networks, Interaction Networks 등입니다.

- 그렇지만, 지금까지는 사실 상 시도 정도에 그쳤고 구조는 더 복잡해졌으며, 결과도 그리 promising 하지 않았던 것이 사실입니다.
지금까지 관계를 추론하기 위한 방법으로는 기호주의(symbolic) 접근이 더 우위를 보여왔습니다. 일단 기호를 정의하고, 이를 논리추론이나 수학적 방식으로 관계를 파악했습니다. 그런데 이 방식은 일단 기호를 정의하는데에서 문제가 발생할 수도 있고 (이를 grounding problem 이라고 합니다) 딱 정해진 논리가 아진 변화하는 상황에 대응하는데 무척이나 어려움을 겪기 때문에 현실에서는 적용하기 어려운 부분들이 많았지요. 그리고 현재 유행하는 딥러닝이나 통계학적 학습을 통해 관계를 파악하려던 접근 방식은 데이터가 적은 경우에는 활용자체가 어렵고, 데이터가 꽤 많이 있어도 복잡한 관계를 저치하는 데에는 근본적인 문제를 드러내고 있었습니다. 현재의 이런 문제를 깔끔히 해결한 접근방법이 바로 Relational Networks 입니다.

- 무엇보다 CNN, LSTM, 구조화된 state description, 자연어 등을 동일한 spatial object 로 다룰 수 있도록 일반화했고, 이들 요소들의 관계를 MLP 기반의 네트워크로 표현하고 학습할 수 있도록 해서 관계의 추정, object 전체에 대한 관계 (세부 순서에 관계없이), 데이터 효율이라는 3마리 토끼를 모두 잡고 있고, 비교적 일반화된 구조를 제시했기 때문에 향후 응용이 엄청나게 나올 수 있을 것 같습니다.

- 아무리 이론이 훌륭해도 실제로 결과가 나쁘면 인정받을 수 없지요. 결과에 대해서도 가장 어려웠던 VQA 관련한 작업인 CLEVR 에서 (기존 최고기록이 68.5% 정답률 (2등이 52.3% 였으니 꽤 큰 격차입니다), 인간에게 시키면 92.6% 맞추는 결과를 넘어선 95.5% 라는 어마어마한 정답률을 만들어냈고, 자연어 기반 QA 인 bAbi, 심지어는 표면에서 다양하게 공들이 움직이는 상황에서 공들의 다양한 관계를 파악하는 작업, 보지 못한 모션캡처에 적용해서 걷는 인간의 관절간의 관계까지 정확하게 예측해내는 등 적용분야도 다양하게 할 수 있음을 입증했습니다 (어찌보면 당연합니다. general 하게 적용할 수 있도록 CNN, LSTM, state description 등을 모두 동일한 object 로 간주할 수 있으니).

- 여튼 또 하나의 인공지능 기술 발전의 전기가 마련된 듯 하네요. 인공지능이 creative 한 작업을 할 수 없다는 선입견을 깬 GAN에 이어서 복잡한 관계를 파악하는 일을 인간보다 잘 할 수 있다는 RN의 등장이 한 동안 빅이슈가 될 것 같습니다


- 연관된 자료들
스탠포드와 FAIR이 발표한,
구성 언어 및 초등 시각 추론을 위한 진단 데이터 세트(CLEVR)
http://cs.stanford.edu/people/jcjohns/clevr/
https://arxiv.org/pdf/1612.06890.pdf
와 관련연구 : 시각 추리를 위한 프로그램 추론 및 실행
http://cs.stanford.edu/people/jcjohns/iep/
https://arxiv.org/pdf/1705.03633.pdf
https://github.com/facebookresearch/clevr-iep
그리고,
순수한 텍스트기반 QnA 데이터세트인 페이스북의 bAbI에 관한 것
https://research.fb.com/downloads/babi/
https://github.com/facebook/bAbI-tasks
입니다.
이들은 딥마인드의 관계추론 논문에서 연구소재로 삼고 있습니다. 이에, 논문
https://arxiv.org/pdf/1706.01427.pdf
을 보다 폭넓게 이해하시고, 후속연구를 실행하시는데 동기와 도움이 되셨으면 하는 바람에서 글을 올립니다.
( 되돌아 보니, 보충자료의 50%정도도 이미 정 교수님께서 5월 중순경에 선보이셨던 내용이네요. 하지만, 아셨던 분들도 이 시점에서 주의 환기용으로 의미 있으리라 생각하고 싶습니다^^; )
