Implementation of Imitaiton leraning models on UR10e available in IRIS Lab, University of Aveiro

This is done using Lerobot interface, so clone lerobot git and follow the instructions given in the github to install lerobot and clone this repository as well

The motivation of this project is to implement and test the Pi-0 model on UR10e available in the laboratory

**First stage**
Implementing different policies like ACT, TDMPC on gym-xarm environemnt as that is the closest to the robot in the laboratory and is well documented 

- script which can record the dataset through teleoperation using joystick and pushes the dataset to hugging face. you can check the datasets made in huggingface id - nik658 


**Second stage**
Implementing rtde control and teleoperation of UR10e using joystick and interfacing with kinect

**Third stage**
Implementing a policy to run locally on my laptop using rtde and lerobot interface which control the real robot

- script to record epsiodes through teleoperation using joystick and push to github
- lerobot train script to train using the huggingface repo where the episodes are pushed
- script to run inference locally

The problem here was that my gpu was not able to locally run inference of pi0 and cpu was too slow

**Fourth stage**
Implemeneting a socket connection between the cloud gpu cluster and the local pc to run the policy on the cluster and handle observation and control robot locally
