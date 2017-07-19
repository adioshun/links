NVIDIA DIGITS 5 (updated Feb 1, 2017)
NVIDIA DIGITS 4 (updated January 20, 2017)
NVIDIA DIGITS 3 (updated Feb 10, 2016)
NVIDIA DIGITS 2 (updated Sept 8, 2015)
NVIDIA DIGITS 1 (updated June 26, 2015)



### 12.1 v3.0
```
# ML_REPO_PKG=nvidia-machine-learning-repo_4.0-2_amd64.deb
# wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/$ML_REPO_PKG
# dpkg -i $ML_REPO_PKG
# apt-get update
# apt-get install digits
```
> 이것으로 Caffe framework까지 한꺼번에 설치됩니다. , [Getting Started](https://github.com/NVIDIA/DIGITS/blob/digits-3.0/docs/GettingStarted.md)

### 12.2 v4.0 ( Object detection 지원)
- [참고](http://m.blog.daum.net/ocean2nd/1764052)

### 12.3 v5.1 (Image Segmentation 지원)

- [Ubuntu Installation](https://github.com/NVIDIA/DIGITS/blob/digits-5.0/docs/UbuntuInstall.md)
- [Getting Started](https://github.com/NVIDIA/DIGITS/blob/digits-5.0/docs/GettingStarted.md)
- [참고](https://www.slideshare.net/pirahansiah/how-to-install-digits-51-on-ubuntu-14)




```
# CUDA 8 for Ubuntu1604 x86
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb

# CUDA 8 for Ubuntu1404 x86
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb`
sudo apt-get update
sudo apt-get install cuda



# DIGITS v5 for CUDA 8, Ubuntu1604 x86
wget https://developer.nvidia.com/compute/machine-learning/digits/secure/5.0/prod/nv-deep-learning-repo-ubuntu1604-ga-cuda8.0-digits5.0_1-1_amd64-deb 

# DIGITS v5 for CUDA 8, Ubuntu1404 x86
wget https://developer.nvidia.com/compute/machine-learning/digits/secure/5.0/prod/nv-deep-learning-repo-ubuntu1404-ga-cuda8.0-digits5.0_2-1_amd64-deb

sudo dpkg -i nv-deep-learning-repo-ubuntu1404-ga-cuda8.0-digits5.0_2-1_amd64-deb
sudo apt-get update
sudo apt-get install DIGITS
```

### 12.3 참고 자료 
[참고 0] DIGITS 4 설치과정 요약  https://github.com/NVIDIA/DIGITS/blob/v4.0.0/README.md
[참고 1] 1. Ubuntu 14.04 설치 https://github.com/NVIDIA/DIGITS/blob/digits-4.0/docs/UbuntuInstall.md
[참고 2] 2. Digits 설치(Caffe 설치 포함)  https://github.com/NVIDIA/DIGITS/blob/v4.0.0/docs/BuildDigits.md 
[참고 2-1] 2-1. Caffe 설치 https://github.com/NVIDIA/DIGITS/blob/v4.0.0/docs/BuildCaffe.md
[참고 3] 예제 파일 설치 및 실행  https://github.com/NVIDIA/nccl
[참고 4] DIGITS 4 예제 - DetectNet https://github.com/NVIDIA/DIGITS/tree/master/examples/object-detection



- Docker를 이용하여 DIGITS 설치하기 : [Docker Hub](https://hub.docker.com/r/nvidia/digits/), [설명](https://github.com/NVIDIA/nvidia-docker/wiki/DIGITS)


[Image Segmentation Using DIGITS 5](https://devblogs.nvidia.com/parallelforall/image-segmentation-using-digits-5/)

[Deep Learning for Object Detection with DIGITS](https://devblogs.nvidia.com/parallelforall/deep-learning-object-detection-digits/)