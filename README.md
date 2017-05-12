# Deeplearning을 공부하기 위해 필요한 자료들 모음 입니다.

88
# 논문

- [Mendeley](https://www.mendeley.com/library/): 논문/서지 관리

- [Overleaf](https://www.overleaf.com/): LaTex 웹기반 작성 툴, [[수식 작성]](http://www.hostmath.com/)

- [Arxiv-sanity](http://www.arxiv-sanity.com/): arxiv사이트 Re-Format 사이트  

- [GitXiv](http://www.gitxiv.com/) : Arxiv사이트 + 오픈소스

- [Distill](http://distill.pub/): Web기반 저널

- [Research Gate](https://www.researchgate.net/home): 연구원들의SNS

- [Fusemind](http://fusemind.org): Google Scholar Chrome 확장 앱

# GitHub 연동

- [GitHub](https://github.com/adioshun): GitHub사이트

- [Prose](http://prose.io/#adioshun): Github Markdown Editor

- [GistBox](https://app.gistboxapp.com/library/my-gists): GIst 코드 모음

- [GitBook](https://www.gitbook.com/@adioshun): Github기반 eBook 작성 툴, [[빈줄 제거]](http://textmechanic.com/text-tools/basic-text-tools/addremove-line-breaks/), [[테이블]](http://truben.no/table/)

- [Insight.io](https://insight.io/account/projects): Githib기반 코드 뷰어



├── evaluation.py # evaluation.py
├── images # model architectures
│   ├── resnet.png
│   ├── vggnet5.png
│   └── vggnet.png
├── MNIST # mnist data (not included in this repo)
│   ├── t10k-images-idx3-ubyte.gz
│   ├── t10k-labels-idx1-ubyte.gz
│   ├── train-images-idx3-ubyte.gz
│   └── train-labels-idx1-ubyte.gz
├── model # model weights
│   ├── resnet.h5
│   ├── vggnet5.h5
│   └── vggnet.h5
├── model.py # base model interface
├── README.md
├── utils.py # helper functions
├── resnet.py
├── vgg16.py
└── vgg5.py
