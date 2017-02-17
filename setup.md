#
# Windows 기반

### R

* R Core 설치 : [https:\/\/cloud.r-project.org\/bin\/windows\/base\/](https://cloud.r-project.org/bin/windows/base/)
* R Studio 설치 : [https:\/\/www.rstudio.com\/products\/rstudio\/download\/](https://www.rstudio.com/products/rstudio/download/)

### Python

* Anaconda 설치 : [https:\/\/www.continuum.io\/downloads](https://www.continuum.io/downloads)

## Linux 기반

### R 분석환경

#### 1. R 설치

```bash
sudo apt-get update
sudo apt-get install r-base
```

#### 2. R-Studio 설치

```bash
$ sudo apt-get install gdebi-core
$ wget https://download2.rstudio.org/rstudio-server-1.0.136-amd64.deb
$ sudo gdebi rstudio-server-1.0.136-amd64.deb
```

> 접속 확인 `http://localhost:8787/`

### Python\/Jupyter 설치

#### 1. Python설치 \(/w conda\)

```bash
wget https://repo.continuum.io/archive/Anaconda3-4.3.0-Linux-x86_64.sh
bash Anaconda3-4.3.0-Linux-x86_64.sh
export PATH="/home/username/anaconda/bin:$PATH" OR 설치 설정시 pretend하기
```

> [https://www.continuum.io/downloads](https://www.continuum.io/downloads) 에서 최신 버전 확인 가능


#### 2. Python 용 R 설치 \(/w conda\)

```bash
conda install -c r r-irkernel
conda install -c r r-essentials
```

###### ipython에서 R 패지키 설치 방법 \(R 콘솔에서 실행??\)

sudo apt-get install libcurl4-openssl-dev 
sudo apt-get install libssl-dev 
```
#R쉘진입
install.packages(c('repr', 'IRdisplay', 'evaluate', 'crayon', 'pbdZMQ', 'devtools', 'uuid', 'digest'))
devtools::install_github('IRkernel/IRkernel')
IRkernel::installspec()
```

> 패키지설치시: install.packages("ldavis", "/home/user/anaconda3/lib/R/library")
> [메뉴얼필독](https://www.r-bloggers.com/jupyter-and-r-markdown-notebooks-with-r/amp/)




#### 3. OpenCV 설치 \(/w conda\)

`conda install -c https://conda.binstar.org/menpo opencv3`

> import cv2 \(!!!IMPORTANT it’s still cv2 not cv3\).
> `To check the version print(cv2.__version__)`

#### 4. Tensorflow 설치 \(/w conda\)

Create a conda environment

```
$ conda create -n tensorflow python=x.x
```

가상공간 활성화 & 설치

```
$ source activate tensorflow
(tensorflow)$ conda install -c conda-forge tensorflow
# Linux/Mac OS X, Python 2.7/3.4/3.5, CPU only:
```

####### pip을이용하여 설치할 경우 가상공간 진입후 하기 ([python3)
```
$ source activate tensorflow
$ (tensorflow)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp34-cp34m-linux_x86_64.whl
$ (tensorflow)$ pip3 install --ignore-installed --upgrade $TF_BINARY_URL
```

###### 3.1 Jupyter 설정하기 \(/w conda, /w tensorflow\)

설치하기

```
$ source activate tensorflow
(tensorflow)$ conda install ipython
(tensorflow)$ conda install jupyter
```

설정하기

```
$ jupyter notebook --generate-config
$ vi /root/.jupyter/jupyter_notebook_config.py
$ nohup jupyter notebook &
```

###### 설정 파일 \# /home/\(username\)/.jupyter/jupyter\_notebook\_config.py
```
c = get\_config\(\)
c.NotebookApp.ip = '\*'
c.NotebookApp.open\_browser = False \# 원격접속으로 활용할 것이기 때문에 비활성화 시켰다.
c.NotebookApp.port = 8017 \# 포트를 설정해준다. 기본포트로 8888이 자동 배정된다.
c.NotebookApp.password = '....' # python 실행후 from notebook.auth import passwd; passwd\(\)
c.NotebookApp.notebook\_dir = '/home/winterj/notebook' \# 기본 디렉터리를 지정시켜준다.
c.NotebookApp.base\_url = 'notebook' \#외부 접근을 위한 필수 작업
```

> 공식 TensorFlow 설치 [[메뉴얼]](https://www.tensorflow.org/versions/master/get_started/os_setup), [[Ref]](http://b.winterj.me/220858584491)

#### Jupyter Lab설치

```
# you will need jupyter notebook >= v4.2
pip3 install jupyterlab
jupyter serverextension enable --py jupyterlab --sys-prefix
jupyter lab
```

#### OpenAI Gym
```

sudo apt install cmake
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
sudo -H pip install gym
sudo -H pip install gym[atari]
```
