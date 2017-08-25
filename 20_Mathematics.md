

# Aritle (blog)

- [[추천] 머신러닝중요 Notation](http://www.deeplearningbook.org/contents/notation.html): 4page, 이것들만 잘 기억하고 있어도 논문에 나오는 수학의 50%는 먹고 들어가는 정말 많이 나오는 기호들입니다.

- [[추천]Explained Visually](http://setosa.io/ev/): 수학 기본 개념을 interactive하게 visualization해주는 사이트

- [How do I learn mathematics for machine learning?](https://www.quora.com/How-do-I-learn-mathematics-for-machine-learning): 확률, 선형대수, 미적분, 행렬대수, 최적화등, 

- [ML notes: Why the log-likelihood?](https://blog.metaflow.fr/ml-notes-why-the-log-likelihood-24f7b6c40f83)

* [마코프: 큰수의 법칙, 베르누이 과정, 마코프 과정](http://fbsight.com/t/topic/43287) : 김성훈 교수 추천 강좌, [[Youtube]](https://youtu.be/Ws63I3F7Moc)

- [Basic Probability](http://students.brown.edu/seeing-theory/basic-probability/index.html#first): 이미지로 쉽게 설명

- [[번역] 머신러닝 속 수학](https://mingrammer.com/translation-the-mathematics-of-machine-learning)

- [Wasserstein GAN 수학 이해하기 I](https://www.slideshare.net/ssuser7e10e4/wasserstein-gan-i)

- [확률(Probability) vs 가능도(Likelihood)](http://rstudio-pubs-static.s3.amazonaws.com/204928_c2d6c62565b74a4987e935f756badfba.html)

- [Linear algebra cheat sheet for deep learning](https://medium.com/towards-data-science/linear-algebra-cheat-sheet-for-deep-learning-cd67aba4526c)
s

- [Essence of linear algebra - 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab): Youtube, 5~15분 이내의 15개 짧은 강좌, 이상화 교수 강의 후 복습 개념으로 보기 추천

- [math monk](https://www.youtube.com/user/mathematicalmonk?app=desktop): measure theory based 확률론을 공부하는걸 추천

# 동영상

- 벡터미적분학 13화. 다변수벡터함수의 다이버전스,커얼 : [1강](https://www.youtube.com/watch?v=jLWjtWWb0I8), [2강](https://www.youtube.com/watch?v=fvtjqkf4Wl4&feature=push-u&attr_tag=_9_gWW3ThBZSGlki-6)

# Tutorial

# 추천 Mooc 커리큘럼

- 함수해석학(Functional analysis - krezig, convex optimization - stephen boyd

- 확률론 (Probabilistic systems, John Tsitsiklis)

- 확률적 그래프모델 (probabilistic graphical model - Daphne Koller)

- Information theory - David Mackay



저처럼 머신러닝,딥러닝 입문해 보시는 분들께 자그마한 정보가 될까 하여 조금 길게 포스팅 해 봅니다. 제가 6개월 정도 달려보았는데요. [[원문]](https://www.facebook.com/groups/TensorFlowKR/permalink/485458151795248/)

###### 1. 처음 시작은 [김성훈 교수님 유튜브 강의](https://www.youtube.com/playlist?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm)로 입문하시는게 최고의 선택인 듯 합니다. 쉽게 풀어주시면서 직관적인 이해를 도와주시는 부분이 참 많습니다.

###### 2. 더 진도를 나가려면 기초를 다지는 것이 좋았습니다.아래 2개 강좌는 반드시 듣는게 매우 중요할 듯 합니다.
  - [선형대수학 강좌](https://www.youtube.com/playlist?list=PLSN_PltQeOyjDGSghAf92VhdMBeaLZWR3): 이상화 교수, [[교재]](http://www.kocw.net/home/search/kemView.do?kemId=977757)
  - [확률통계 강좌](https://www.youtube.com/playlist?list=PLSN_PltQeOyjmRIsC7VNirXOBqWoypd4V): 이상화 교수

> 위 2개 강의를 듣고 나면, 신경망에 대한 조금은 본질적인 개념 이해나 차원축소 기법들 이해하는데 도움이 되고 수식 notation이 눈에 많이 들어오는 것 같았습니다. 그리고 카이스트 문일철 교수님 강의도 좋은데, 바로 들어가면 어렵더군요. 위 3개 강좌를 듣고 들어가시는 게 아마도 좋지 않을까 합니다. 그리고 문교수님 강의 듣기 전에 하나 더 선행하면 좋을 듯 합니다.

###### 3. 충북대 이건명 교수님 강의 중에서 ‘[탐색과 최적화](http://www.kocw.net/home/search/kemView.do?kemId=1170523)’ 부분을 듣고 가시는게 좋아 보입니다.위 4개 강좌를 듣고 나면, 윤곽이 잡힙니다.

###### 4. 이렇게 해서 KOOC에 있는 카이스트 [문일철 교수님 강의](http://seslab.kaist.ac.kr/xe2/page_GBex27)는 마지막에 들으면 이제는 많이 다가오는 것 같습니다. [[Youtube]](https://www.youtube.com/channel/UC9caTTXVw19PtY07es58NDg)
  - k-means, gmm(gaussian mixture model), em(expectation & maximization) 강의를 들으면 variational inference의 기초가 잡힐 듯 합니다.
  - hmm(hidden markov model) 강의를 들으면 rnn이 생각나고 또한 bidirectional rnn이 생각나고, 해당 강의에서 dynamic programming 얘기를 들으면 attention model이 생각날 것으로 보입니다.
  - 샘플링 쪽을 들으면 mcmc 기초가 잡히고 gibbs sampling을 들으면 RBM이 생각날 듯 합니다.

###### 5. 이렇게 하고 TF-KR에서 진행하는 PR12 논문 리뷰 동영상을 보면, 이제는 많이 친숙하게 다가오지 않을까 합니다.

> 아… 그리고 문교수님 강의 들으면서 앤드류 교수님 강의 병행하면 시너지가 많이 날 듯 합니다.
