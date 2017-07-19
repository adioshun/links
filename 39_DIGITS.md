NVIDIA DIGITS 5 (updated Feb 1, 2017)
NVIDIA DIGITS 4 (updated January 20, 2017)
NVIDIA DIGITS 3 (updated Feb 10, 2016)
NVIDIA DIGITS 2 (updated Sept 8, 2015)
NVIDIA DIGITS 1 (updated June 26, 2015)


# 설치  

## 0. GPU지원 

- [CUDA설치](https://github.com/adioshun/Blog_Jekyll/blob/master/2017-07-18-CUDA_CuDNN_Installation.md)

## 1. DIGITS 5 설치 

#### 1.1 Apt-get로 설치 

```
# For Ubuntu 14.04
CUDA_REPO_PKG=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
ML_REPO_PKG=http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/nvidia-machine-learning-repo-ubuntu1404_4.0-2_amd64.deb

# For Ubuntu 16.04
CUDA_REPO_PKG=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
ML_REPO_PKG=http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb

# Install repo packages
wget "$CUDA_REPO_PKG" -O /tmp/cuda-repo.deb && sudo dpkg -i /tmp/cuda-repo.deb && rm -f /tmp/cuda-repo.deb
wget "$ML_REPO_PKG" -O /tmp/ml-repo.deb && sudo dpkg -i /tmp/ml-repo.deb && rm -f /tmp/ml-repo.deb

# Download new list of packages
sudo apt-get update

sudo apt-get install digits
```

> [Ubuntu Installation](https://github.com/NVIDIA/DIGITS/blob/digits-5.0/docs/UbuntuInstall.md)

#### 1.2 deb로 설치 
```
# DIGITS v5 for CUDA 8, Ubuntu1604 x86
wget https://developer.nvidia.com/compute/machine-learning/digits/secure/5.0/prod/nv-deep-learning-repo-ubuntu1604-ga-cuda8.0-digits5.0_1-1_amd64-deb 

# DIGITS v5 for CUDA 8, Ubuntu1404 x86
wget https://developer.nvidia.com/compute/machine-learning/digits/secure/5.0/prod/nv-deep-learning-repo-ubuntu1404-ga-cuda8.0-digits5.0_2-1_amd64-deb

sudo dpkg -i nv-deep-learning-repo-ubuntu1404-ga-cuda8.0-digits5.0_2-1_amd64-deb
sudo apt-get update
sudo apt-get install DIGITS
```

### 1.3 Docker로 설치 
- Docker를 이용하여 DIGITS 설치하기 : [Docker Hub](https://hub.docker.com/r/nvidia/digits/), [설명](https://github.com/NVIDIA/nvidia-docker/wiki/DIGITS)

# 실행 

확인 : sudo service digits status

접속 : http://localhost 

설정(포트): sudo dpkg-reconfigure digits

> [Getting Started](https://github.com/NVIDIA/DIGITS/blob/digits-5.0/docs/GettingStarted.md)








[Image Segmentation Using DIGITS 5](https://devblogs.nvidia.com/parallelforall/image-segmentation-using-digits-5/)

[Deep Learning for Object Detection with DIGITS](https://devblogs.nvidia.com/parallelforall/deep-learning-object-detection-digits/)