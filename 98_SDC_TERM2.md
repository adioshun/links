# Term2

## 1. Sensor Fusion
- Sensors
The first lesson of the Sensor Fusion Module covers the physics of two of the most import sensors on an autonomous vehicle — radar and lidar.

- Kalman Filters
Kalman filters are the key mathematical tool for fusing together data. Implement these filters in Python to combine measurements from a single sensor over time.

- C++ Primer
Review the key C++ concepts for implementing the Term 2 projects.
  - Project: Extended Kalman Filters in C++
Extended Kalman filters are used by autonomous vehicle engineers to combine measurements from multiple sensors into a non-linear model. Building an EKF is an impressive skill to show an employer.

- Unscented Kalman Filter
The Unscented Kalman filter is a mathematically-sophisticated approach for combining sensor data. The UKF performs better than the EKF in many situations. This is the type of project sensor fusion engineers have to build for real self-driving cars.

> Project: Pedestrian Tracking / Fuse noisy lidar and radar data together to track a pedestrian.

## 2. Localization
- Motion
Study how motion and probability affect your belief about where you are in the world.

- Markov Localization
Use a Bayesian filter to localize the vehicle in a simplified environment.

- Egomotion
Learn basic models for vehicle movements, including the bicycle model. Estimate the position of the car over time given different sensor data.

- Particle Filter
Use a probabilistic sampling technique known as a particle filter to localize the vehicle in a complex environment.

- High-Performance Particle Filter
Implement a particle filter in C++.

> Project: Kidnapped Vehicle / Implement a particle filter to take real-world data and localize a lost vehicle.

## 3. Control
- Control
Learn how control systems actuate a vehicle to move it on a path.

- PID Control
Implement the classic closed-loop controller — a proportional-integral-derivative control system.

- Linear Quadratic Regulator
Implement a more sophisticated control algorithm for stabilizing the vehicle in a noisy environment.

> Project: Lane-Keeping / Implement a controller to keep a simulated vehicle in its lane. For an extra challenge, use computer vision techniques to identify the lane lines and estimate the cross-track error.


- [Tracking pedestrians for self driving cars](https://medium.com/towards-data-science/tracking-pedestrians-for-self-driving-cars-ccf588acd170)

# Project 6 — Extended Kalman Filter

- [Helping a Self Driving Car Localize itself](https://medium.com/@priya.dwivedi/helping-a-self-driving-car-localize-itself-88705f419e4a): Priya Dwivedi, Particle Filters: higher accuracy — less than 10 cm, [[GitHUb]](https://github.com/priya-dwivedi/CarND-Kidnapped-Vehicle-Project)

- [Robot Localization using Particle Filter](https://medium.com/@ioarun/robot-localization-using-particle-filter-fe051c5d38e2): Arun Kumar

- [Udacity Self-Driving Car Nanodegree Project 6 — Extended Kalman Filte](https://medium.com/udacity/udacity-self-driving-car-nanodegree-project-6-extended-kalman-filter-c3eac16c283d): Jeremy Shannon

# pedestrians
- [Tracking pedestrians for self driving cars](https://medium.com/towards-data-science/tracking-pedestrians-for-self-driving-cars-ccf588acd170): Priya Dwivedi
