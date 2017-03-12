![](https://cdn-images-1.medium.com/max/800/1*HTyatXEYA62kjN_kNxYW7g.png)

[2017: The year for autonomous vehicles](https://machinelearnings.co/2017-the-year-for-autonomous-vehicles-8359fec2d2db#.bh2nr1v38)

# 0. 개요

## Self-Driving Engineer
* [How to Become a Self-Driving Car Engineeer Talk](https://medium.com/self-driving-cars/how-to-become-a-self-driving-car-engineeer-talk-923dfa5e6665#.ig03r4tmq) : 추천, ppt,Jupyter코드 포함
* [But, Self-Driving Car Engineers don’t need to know C/C++, right?](https://medium.com/@mimoralea/but-self-driving-car-engineers-dont-need-to-know-c-c-right-3230725a7542#.1pk5qsb90) : 필요 지식 및 기술(개인 의견)
* [Self Driving Car Engineer Deep Dive](https://medium.com/@paysa/self-driving-car-engineer-deep-dive-89b814f3ff04#.pygljklaq)
* [Who’s Hiring Autonomous Vehicle Engineers](https://medium.com/self-driving-cars/whos-hiring-autonomous-vehicle-engineers-1ccf42185e08#.zd2odb5gs)
- [Five Skills Self-Driving Companies Need](https://hackernoon.com/five-skills-self-driving-companies-need-8546d2aba7c1#.bt1p1a7jq)
- <Del>[5 Things That Give Self-Driving Cars Headaches](https://getpocket.com/a/read/1625729922): 예측 불가 인간, 날씨, 우회길, 웅덩이 </Del>

## Youtube
- [16 Questions About Self-Driving Cars](https://vimeo.com/198256576)[[Q List]](https://medium.com/self-driving-cars/frank-chens-16-questions-about-self-driving-car-3c663987965b#.85q8lxbdy)
- [Autonomous Vehicles Overview](https://www.youtube.com/watch?v=CruCp6vqPQs&feature=youtu.be) : Wiley Jones,2016. 8. 28, 56분, Robotics, actuation, sensors, SLAM, computational platforms

## 논문
* [논문: End to End Learning for Self-Driving](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) : NVIDIA 2016 Paper
* [논문: End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316) : 카메라 3대와 운전자의 핸들 조작+알파로 학습한다음 카메라 하나만 입력으로 사용하고 운전대를 어떻게 움직이는지를 예측하여 자동운전,  [[YOUTUBE]](https://drive.google.com/file/d/0B9raQzOpizn1TkRIa241ZnBEcjQ/view)




# 1. Udacity’s Self-Driving Car
* Udacity제공 자율 주행 관련 교육 프로그램[[홈페이지]](https://www.udacity.com/drive), [[GitHub: Code]](https://github.com/udacity/self-driving-car), [[GitHub: 시뮬레이션]](https://github.com/udacity/self-driving-car-sim)), [[Slack]](https://nd013.slack.com/messages/@slackbot/)
  * [Term1 커리큘럼](https://medium.com/self-driving-cars/term-1-in-depth-on-udacitys-self-driving-car-curriculum-ffcf46af0c08#.9kzwdddso)
  * [Term2 커리큘럼](https://medium.com/udacity/term-2-in-depth-on-udacitys-self-driving-car-curriculum-775130aae502#.drwc2n71n)
  * Term 2 강의 신청 (by 2017.04.25) [[링크]](https://admissions.udacity.com/new/nd013)
  * [Challenges : We’re Building an Open Source Self-Driving Car](https://medium.com/udacity/were-building-an-open-source-self-driving-car-ac3e973cd163#.w3hnxhv6l)

> [GitHub: Free Nanodegree Program](https://github.com/mikesprague/udacity-nanodegrees)

## 1. Challenges
[List of Challenges](https://www.udacity.com/self-driving-car)

### 1.1 Challenge #1 : 3D Model for Camera Mount
* [개요 설명](https://medium.com/udacity/challenge-1-3d-model-for-camera-mount-f5ffcc1655b5#.gag46u20q)

> 기기 제작 관련 내용이라 Skip

### 1.2 Challenge #2 : Using Deep Learning to Predict Steering Angles.
* [개요 설명](https://medium.com/udacity/challenge-2-using-deep-learning-to-predict-steering-angles-f42004a36ff3#.7vo2x0vfn),  [결과 설명](https://medium.com/udacity/teaching-a-machine-to-steer-a-car-d73217f2492c#.84hhfrsfc)
  * [GitHub: End to End Learning for Self-Driving Car](https://github.com/windowsub0406/Behavior-Cloning)
  * [Coding a Deep Neural Network to Steer a Car: Step By Step](https://medium.com/udacity/coding-a-deep-neural-network-to-steer-a-car-step-by-step-c075a12108e2#.7lxkxhoyd)
* 1등 : Team Komanda(Team Lead: Ilya Edrenkin), [[GitHub]](https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/komanda)
* 2등 : Team Rambo(Team Lead: Tanel Pärnamaa), [[GitHub]](https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/rambo)
* 3등 : Team Chauffeur(Team Lead: Matt Forbes),[간략 설명](https://medium.com/udacity/coding-a-deep-neural-network-to-steer-a-car-step-by-step-c075a12108e2#.pmiaer9k0),
 [[GitHub]](https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/chauffeur)

### 1.3 Challenge #3: Image-Based Localization
* [개요 설명](https://medium.com/udacity/challenge-3-image-based-localization-5d9cadcff9e7#.3lstquvvf)

### 1.4 Challenge #4: Self-Driving Car Android Dashboard
* [개요 설명](https://medium.com/udacity/challenge-4-self-driving-car-android-dashboard-83a2a5c8b29e#.1ndw38fam)


## 2 Projects
[Daniel Stang](http://www.cellar--door.com/sdc) : Project 1~5까지의 해결 과정/내용 정리

### 2.1 Project 1 — Detect Lane Lines
- <del>[Building a lane detection system using Python 3 and OpenCV](https://medium.com/@galen.ballew/opencv-lanedetection-419361364fc0#.c2gg3agft) : Galen Ballew [[GitHub]](https://github.com/galenballew/Lane-Detection-OpenCV)</del>
- <del>[My Lane Detection Project in the Self Driving Car Nanodegree by Udacity](https://medium.com/@paramaggarwal/my-lane-detection-project-for-the-self-driving-car-nanodegree-by-udacity-36a230553bd3#.iy3xxtun9) : Param Aggarwal</del>
- [Bugger! Detecting Lane Lines](http://www.jessicayung.com/bugger-detecting-lane-lines/) : Jessica Yung
- [Hello Lane Lines](http://www.blog.autonomousd.com/2016/12/hello-lane-lines.html?showComment=1485840449048#c2413863224281126452) : Josh Pierro

### 2.2 Project 2- Traffic Sign Classifier
- <del>[How to identify a Traffic Sign using Machine Learning !!](https://medium.com/@sujaybabruwad/how-to-identify-a-traffic-sign-using-machine-learning-7aa98c871469#.12d91nj06) : Sujay Babruwad</del>
- <del>[Traffic Sign Classification](https://medium.com/@hengcherkeng/updated-my-99-40-solution-to-udacity-nanodegree-project-p2-traffic-sign-classification-5580ae5bd51f#.iobd79qzu) : Cherkeng Heng, [[UPdated]](https://medium.com/@hengcherkeng/updated-my-99-68-solution-to-udacity-nanodegree-project-p2-traffic-sign-classification-56840768dec8#.lg10dmxg7)</del>
- <del>[Traffic Sign Classifier: Normalising Data](http://www.jessicayung.com/traffic-sign-classifier-normalising-data/) : 전처리 부분 중심으로, Jessica Yung</del>

#### [다른 소스에서의 자료 ]
- Traffic Sign Recognition with TensorFlow
  - [확인] [Part1](https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6#.v2g5b44hl) : Image classification with Simple model  [[GitHub]](https://github.com/waleedka/traffic-signs-tensorflow/blob/master/notebook1.ipynb)
  - Part2() : 아직 작성 안됨,  Convolutional networks, data augmentation, and object detection.
- [Kaggle : Traffic Sign Recognition](https://inclass.kaggle.com/c/traffic-sign-recognition)

### 2.3 Project 3 — Behavioral Cloning
* <del>[Udacity Self-Driving Car Nanodegree Project 3 — Behavioral Cloning](https://medium.com/udacity/udacity-self-driving-car-nanodegree-project-3-behavioral-cloning-446461b7c7f9#.j6t0algy9) : Jeremy Shannon, 핸들 각도 분포에 따른 성능 평가 </del>  [[GitHub: 기술적 설명]](https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project)
* <del>[Teaching a car to drive itself](https://chatbotslife.com/teaching-a-car-to-drive-himself-e9a2966571c5#.g851pul4b) Arnaldo Gunzi, 전처리의 모든 부분 커버 </del>
- <del>[Training a Self-Driving Car via Deep Learning](http://blog.openroar.com/2016/12/29/self-driving-car-deep-learning/) : James Jackson </del>
- [GitHub: Behavioral Cloning](https://github.com/jinchenglee/CarND-Behavioral-Cloning/blob/master/README.md) : jinchenglee
- [Behavioural Cloning Applied to Self-Driving Car on a Simulated Track](https://medium.com/towards-data-science/behavioural-cloning-applied-to-self-driving-car-on-a-simulated-track-5365e1082230#.wkv74ptvr)
- [Self Driving Car — Technology drives the Future !!](https://medium.com/@sujaybabruwad/teaching-a-car-to-ride-itself-by-showing-it-how-a-human-driver-does-it-797cc9c2462b#.3a1pznag8)
- [You don’t need lots of data! (Udacity Behavioral Cloning)](https://medium.com/@fromtheast/you-dont-need-lots-of-data-udacity-behavioral-cloning-6d2d87316c52#.lw2tuad26)
- [GitHub: windowsub0406](https://github.com/windowsub0406/Behavior-Cloning)
- <del> [Behavioral Cloning — Transfer Learning with Feature Extraction](https://medium.com/@kastsiukavets.alena/behavioral-cloning-transfer-learning-with-feature-extraction-a17b0ebabf67#.8rw2nug86): Alena Kastsiukavets, Transfer Learning 기법 적용 </del>
- <del>[Denise R. JamesFollowing](https://medium.com/@deniserjames/denise-james-bsee-msee-5beb448cf184#.b08sv8h38) </del>
- [Behavioral Cloning For Self Driving Cars](https://medium.com/@ValipourMojtaba/my-approach-for-project-3-2545578a9319#.337eo3ns7) : Mojtaba Valipour
- [An augmentation based deep neural network approach to learn human driving behavior](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.10nhc95af) : Vivek Yadav
- [
Attempting to Visualize a Convolutional Neural Network in Realtime](https://medium.com/@paramaggarwal/attempting-to-visualize-a-convolutional-neural-network-in-realtime-1edd1f3d6c13#.r2q33ajg7) : Param Aggarwal
- [MainSqueeze: The 52 parameter model that drives in the Udacity simulator](https://mez.github.io/deep%20learning/2017/02/14/mainsqueeze-the-52-parameter-model-that-drives-in-the-udacity-simulator/) : Mez Gebre, 추천
- [End-to-end learning for self-driving cars](http://navoshta.com/end-to-end-deep-learning/) : Alex Staravoitau, 강추
- [Self-Driving Car Simulator — Behavioral Cloning](https://medium.com/@jmlbeaujour/self-driving-car-simulator-behavioral-cloning-p3-c9f4338c86b0#.duindt4b0) : Jean-Marc Beaujour
- [Training a deep learning model to steer a car in 99 lines of code](https://hackernoon.com/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a#.ix1eccm5j): Matt Harvey, Keras코드 제공


### 2.4 Project 4 — Advanced Lane Line Finding
![Line detection](https://cdn-images-1.medium.com/max/800/1*861hYMj2um2xgPzmeaJizQ.png)
- [Udacity Advance Lane Finding Notes](https://medium.com/@fromtheast/computer-vision-resources-411ae9bfef51#.h7wz91uf6) : A Nguyen
- <del>[Advanced Lane Line Project](https://chatbotslife.com/advanced-lane-line-project-7635ddca1960#.is20rudnp) : Arnaldo Gunzi, 추천 </del>
* [Advanced Lane Finding](https://medium.com/@ajsmilutin/advanced-lane-finding-5d0be4072514#.trmrym9py) : Milutin N. Nikolic, 추천
* <Del>[Advanced Lane detection](https://medium.com/@MSqalli/advanced-lane-detection-6a769de0d581#.29pkw239p) : Mehdi Sqalli </del>
- [Udacity SDCND : Advanced Lane Finding Using OpenCV](https://medium.com/@heratypaul/udacity-sdcnd-advanced-lane-finding-45012da5ca7d#.ht4k5r8p0): Paul Heraty
- <del>[Self-Driving Car Engineer Diary — 6](https://medium.com/@andrew.d.wilkie/self-driving-car-engineer-diary-6-15ca3fa08277#.dxkxvqw6e) : Andrew Wilkie, Keras활용 </Del>


### 2.5 Project 5 - Vehicle Detection
* <del>[Vehicle Detection and Distance Estimation](https://medium.com/@ajsmilutin/vehicle-detection-and-distance-estimation-7acde48256e1#.cicseb3pb) : Milutin N. Nikolic,[추천] 차간 거리 인식 포함 </del>
- <del>[Small U-Net for vehicle detection](https://chatbotslife.com/small-u-net-for-vehicle-detection-9eec216f9fd6#.75z3or11a) : Vivek Yadav, U-Net 뉴럴네트워크를 이용한 접근법 </del>
* [Feature extraction for Vehicle Detection using HOG+](https://medium.com/@mohankarthik/feature-extraction-for-vehicle-detection-using-hog-d99354a84d10#.2baik24s1) : 차량 인식 제안 방안 설명
- [Vehicle Detection and Tracking using Computer Vision](https://chatbotslife.com/vehicle-detection-and-tracking-using-computer-vision-baea4df65906#.45tw9zjae) : Arnaldo Gunzi, 위젯을 이용한 기법 소개
- <del>[Vehicle tracking using a support vector machine vs. YOLO](https://medium.com/@ksakmann/vehicle-detection-and-tracking-using-hog-features-svm-vs-yolo-73e1ccb35866#.79czobkvx) : Kaspar Sakmann, svm과 YOLO비교 </del>

### 2.6 강사 코멘트
- [Behavioral Cloning](https://medium.com/udacity/how-udacitys-self-driving-car-students-approach-behavioral-cloning-5ffbfd2979e5#.utl189or1) : Behavioral Cloning
- [6 Awesome Projects from Udacity Students](https://medium.com/self-driving-cars/6-awesome-projects-from-udacity-students-and-1-awesome-thinkpiece-550004812558#.q86pa2wnr) : P1~P5 전체
- [6 Different End-to-End Neural Networks](https://medium.com/self-driving-cars/6-different-end-to-end-neural-networks-f307fa2904a5#.m9oak3lui) : Behavioral Cloning
- [Udacity Self-Driving Car Students on Neural Networks and Docker](https://medium.com/self-driving-cars/udacity-self-driving-car-students-on-neural-networks-and-docker-ce6d0e8aa8a5#.h35v6qssx)
- [More Udacity Self-Driving Car Students, In Their Own Words](https://medium.com/self-driving-cars/more-udacity-self-driving-car-students-in-their-own-words-193b99ee66eb#.i967c57v5)
- [Udacity Self-Driving Car Students in Their Own Words](https://medium.com/self-driving-cars/self-driving-car-student-posts-171fcf4cd7a1#.up9x404km)
- [Computer Vision](https://medium.com/self-driving-cars/how-udacity-students-learn-computer-vision-3eefb9d6b552#.ai0w0mwxy)
- [Lane Lines, Curvature, and Cutting-Edge Network Architectures](https://medium.com/self-driving-cars/udacity-students-on-lane-lines-curvature-and-cutting-edge-network-architectures-6deef049659f#.2q7up4bey)
- [Cutting-Edge Autonomous Vehicle Tools](https://medium.com/self-driving-cars/udacity-students-on-cutting-edge-autonomous-vehicle-tools-3c540eb7397f#.iflpootft)
- [Computer Vision, Tiny Neural Networks, and Careers](https://medium.com/self-driving-cars/udacity-students-on-computer-vision-neural-networks-and-careers-f6297d9cb15f#.wwiqwhmct)
- [Udacity Students Who Love Neural Networks](https://medium.com/self-driving-cars/udacity-students-who-love-neural-networks-f5ccb0826b0f#.89p4fjkqe)


### 2.7 Tips
- [Introduction to Udacity Self-Driving Car Simulator](https://medium.com/towards-data-science/introduction-to-udacity-self-driving-car-simulator-4d78198d301d#.rit7ljgmr) : 시뮬레이터 설명, Naoki Shibuya
- [CNN Model Comparison in Udacity’s Driving Simulator](https://medium.com/@chrisgundling/cnn-model-comparison-in-udacitys-driving-simulator-9261de09b45#.4ftznxoee) : 2개의 CNN모델 비교, Chris Gundling
- [Finding the right parameters for your Computer Vision algorithm](https://medium.com/@maunesh/finding-the-right-parameters-for-your-computer-vision-algorithm-d55643b6f954#.rl7dj0jxs) : cv알고리즘의 알맞은 파라미터 선정, maunesh
- [What kind of background do you need to get into Machine Learning?](http://www.chaseschwalbach.com/what-kind-of-background-do-you-need-to-get-into-machine-learning/) : Chase Schwalbach
- [Self-driving car in a simulator with a tiny neural network](https://medium.com/@xslittlegrass/self-driving-car-in-a-simulator-with-a-tiny-neural-network-13d33b871234#.uz2xos9up) : Mengxi Wu
- [MiniFlow from Python to JavaScript](https://medium.com/@tempflip/what-ive-learned-about-neural-networks-when-porting-miniflow-from-python-to-javascript-33ef3c143b5c#.88v5wq110)
- [Preparation, Generalization, and Hacking Cars](https://medium.com/self-driving-cars/carnd-0-on-preparation-generalization-and-hacking-cars-41f4f54be5ca#.m00tx4gkr)
- [Hardware, tools, and cardboard mockups](https://djwbrown.github.io/self-driving-nanodegree/brz/update/2017/01/23/hardware-tools-and-cardboard-mockups.html) : Dylan Brown
- [Comparing model performance: Including Max Pooling and Dropout Layers](http://www.jessicayung.com/comparing-model-performance-including-max-pooling-and-dropout-layers/) : Jessica Yung

### 2.7 저자별 모음
P1: Detect Lane Lines
P2: Traffic Sign Classifier
P3: Behavioral Cloning
P4: Advanced Lane Line Finding
P5: Vehecle Detection

- DAVID A. VENTIMIGLIA: P1, P2, [P3](http://davidaventimiglia.com/carnd_behavioral_cloning_part1.html?fb_comment_id=1429370707086975_1432730663417646&comment_id=1432702413420471&reply_comment_id=1432730663417646#f2752653e047148), [P4](http://davidaventimiglia.com/advanced_lane_lines.html), [P5](http://davidaventimiglia.com/vechicle_detection.html)
- Arnaldo Gunzi: P1, P2, [P3](https://chatbotslife.com/teaching-a-car-to-drive-himself-e9a2966571c5#.xhycchtm4), [P4](https://chatbotslife.com/advanced-lane-line-project-7635ddca1960#.twavzh60e), [P5](https://chatbotslife.com/vehicle-detection-and-tracking-using-computer-vision-baea4df65906#.t6aqd290a)
- Milutin N. Nikolic: P1, P2, P3, [P4](https://medium.com/@ajsmilutin/advanced-lane-finding-5d0be4072514#.a47g46ddk), [P5](https://medium.com/towards-data-science/vehicle-detection-and-distance-estimation-7acde48256e1#.23x509ck5)
- Kaspar Sakmann: P1, P2, [P3](https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.96mwidezy), P4, [P5](https://medium.com/@ksakmann/vehicle-detection-and-tracking-using-hog-features-svm-vs-yolo-73e1ccb35866#.cvv7xwg2t)
- Vivek Yadav: P1, [P2](https://chatbotslife.com/german-sign-classification-using-deep-learning-neural-networks-98-8-solution-d05656bf51ad#.pehltjdi2), [P3-1](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.qju3jgn6n),[P3-2](https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.b10ontn6k) [P4](https://chatbotslife.com/robust-lane-finding-using-advanced-computer-vision-techniques-46875bb3c8aa#.9vxso5cqk), [P5-1](https://chatbotslife.com/small-u-net-for-vehicle-detection-9eec216f9fd6#.ys1nu1ovc), [P5-2](https://chatbotslife.com/towards-a-real-time-vehicle-detection-ssd-multibox-approach-2519af2751c#.phldagwx4)
- Andrew Wilkie(Keras활용): [P1](https://medium.com/@andrew.d.wilkie/self-driving-car-engineer-diary-2-11eacba4d2a7#.ucg0skl79), [P2](https://medium.com/@andrew.d.wilkie/self-driving-car-engineer-diary-4-c75150cf93d5#.7yds41cdl), [P3](https://medium.com/@andrew.d.wilkie/self-driving-car-engineer-diary-5-63d2daab4591#.zfartnf4j), [P4](https://medium.com/@andrew.d.wilkie/self-driving-car-engineer-diary-6-15ca3fa08277#.fxotng3xb), P5
- ooo: [P1](), [P2](), [P3](), [P4](), [P5]()

# 2. Nexar
Nexar is a community-based AI dash cam app for iPhone and Android : [홈페이지](https://www.getnexar.com/), [Challenge](https://www.getnexar.com/challenges/)
- You can compete to win prizes (1st place $5,000, 2nd place $2,000, 3rd place iPhone 7)

## Challenge #1 : USING DEEP LEARNING FOR TRAFFIC LIGHT RECOGNITION
[챌리지 개요/요구사항](https://challenge.getnexar.com/challenge-1)
- [Recognizing Traffic Lights With Deep Learning](https://medium.freecodecamp.com/recognizing-traffic-lights-with-deep-learning-23dae23287cc#.l2iu0aqag) : David Brailovsky
- [The world through the eyes of a self-driving car](https://medium.freecodecamp.com/what-is-my-convnet-looking-at-7b0533e4d20e#.wnom5expq) : David Brailovsky

## Challenge #2 : Coming Soon

# 3. commai
- [홈페이지]() [[GitHub]](https://github.com/commaai/research)
- [논문](http://arxiv.org/abs/1608.01230) : Learning a Driving Simulator


# Article
- [Self-driving cars in the browser](http://janhuenermann.com/projects/learning-to-drive)
- [Towards a real-time vehicle detection: SSD multibox approach](https://medium.com/@vivek.yadav/towards-a-real-time-vehicle-detection-ssd-multibox-approach-2519af2751c#.cldxjz489) : Vivek Yadav


# Open Data
* [옥스포드 Robot Car Dataset](http://robotcar-dataset.robots.ox.ac.uk/index.html)
* [Comma.ai driving dataset](https://github.com/commaai/research) : 자율 주행, 7.5 hours of camera images, steering angles, and other vehicle data.
* [Traffic Sign](http://www.vision.ee.ethz.ch/~timofter/traffic_signs/) : 신호등 데이터셋
* [BelgiumTS Dataset](http://btsd.ethz.ch/shareddata/index.html) : 도로 안내판 데이터셋
* [Udacity Driving Dataset](https://medium.com/udacity/open-sourcing-223gb-of-mountain-view-driving-data-f6b5593fbfa5#.1aq6pztwj) : Mountain View, 223G(10시간)
- [INI](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) : 독일 신호등 데이터셋
- [Udacity_Annotated Driving Dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) : 도로상 차량, 트럭, 보행자 4.5GB
- [KITTI](http://www.cvlibs.net/datasets/kitti/):
- [GTI](http://www.gti.ssr.upm.es/data/Vehicle_database.html): 차량 이미지 데이터셋

# 장비/센서
- [Lida](https://www.blackmoreinc.com/) : [간략 설명](https://medium.com/self-driving-cars/startup-watch-blackmore-1c0f43e24467#.1lfeyxf5f)
- [Mini Autonomous Vehicle](https://medium.com/self-driving-cars/miniature-autonomous-vehicle-dc48d0740afc#.aa4w6l1bs)

# Lab
- [버클리대 DeepDrive](http://bdd.berkeley.edu) : 선진 연구 분야 살펴 보기 좋음

# Startups
- [NAUTO](https://medium.com/self-driving-cars/startup-watch-nauto-1fc88c00a809#.51ybxti2k)
- [Zoox]](https://medium.com/self-driving-cars/startup-watch-zoox-b99b64a1db30#.nce3tldm7)
