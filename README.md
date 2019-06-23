# Udacity Deep Reinforcment Learning Nanodegree 
## Project 1: Navigation:
 This is my implementation of Navigation project from Udacity NanoDegree program (Deep Reinforcement Learning)

### Introduction

The goal for this project is to train an agent to navigate in a large, square world. In addition to navigation, the agent has to learn to collect yellow bananas, while avoiding the blue ones! -[ScreenShot](./Results/EnvironmentIntro.png)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  

This task is episodic, and the condition to solve the environment is, to collect an average score of +13 or more, over 100 consecutive episodes.

### Setup Instructions:
#### Requirements

To reproduce the results from this repository, it is suggested to use virtual python environment and python version 3.6. Python 3.7 at the point of creating this repository does not support tensorflow=1.7 which is a dependency of unityagents package. Note* Python3.7 can still be used, if you know how to install pacakages from source, change requirements.txt and use latest version of tensorflow(tested with tf-v1.14). Follow these simple steps to:

```
 git clone https://github.com/AkhilSinghRana/NavigateAgent.git
 cd NavigateAgent (cloned Repository root)
 virtualenv env_name -p python3
 source env_name/bin/activate #for linux or env_name\Scripts\activate.bat for Windows.
 pip install -e .
 
 ```
 
The above code will setup all the required dependencies for you. You are now ready to open the ipython notebook and train the navigation agent!

``` jupyter lab/notebook Navigation_Soln_Vector_ObsState.ipynb  ```
 

To train an agent capable of collecting bananas, you should first clone this repo and also download (and locate in the same folder) the unity environment according to the OS you're using. The steps are:

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
   

### Results

The best model found was a DQN agent with Batch Normalization, able to solve the environment in only 234 episodes! This was achieved after a careful hyperparameter tuning, leading to a significant improvement compared to the initial base case settings.

![Results for base case and best model settings](./Results/BaseScores.png)

