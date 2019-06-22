# NavigateAgent
This is my implementation of Navigation project from Udacity NanoDegree program (Deep Reinforcement Learning)

### Udacity Deep Reinforcment Learning Nanodegree 
# Project 1: Navigation

### Introduction

For this project, the goal is to train an agent to navigate (and collect bananas!) in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Files and Folders
* main.py: the main routine for the basic-banana project. It is the high-level calls to construct the environment, agent, and train. 
* dqn.py: the class definitions of the agent implementing a DQN algorithm
* ddqn.py: the class definitions of the agent implementing a double-DQN algorithm
* model.py: the deep neural network model class is defined in this file.
* replay_buffer: the class definition for the replay buffer used in DQN and double-DQN algorithms.
* train.py: training routine
* navigate.py: a routine to be called after the agent was trained to visualize its behaviour.
* run_tests.py : a routine for hyperparameter tuning
* utils.py: includes helper functions for logging, saving parameters, plots, and also to generate trajectories after the model was trained.
* log.txt: contains the logs during hyperparameter tuning
* folder ``report``: contains the report for this project.
* folder ``figures``: contains the figures generated for this project.
* folder ``tests``: contains the hyperparameters, plots and weights for the models during hyperparameter tuning
* results.ipynb: helper notebook for analyzing the results.
 

### How to use this repo

To train an agent capable of collecting bananas, you should first clone this repo and also download (and locate in the same folder) the unity environment according to the OS you're using. The steps are:

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the same folder where you cloned this repo

3. Run main.py in a terminal and pass a test name (Required), for example
```
>> python main.py -t "mytest"
```
There are several arguments that can be passed as optionals, which can be checked by looking into main.py:

```python
parser.add_argument('--buffer_size', default = int(1e6), help='replay buffer size')
parser.add_argument('--batch_size', default = 64, help='minibatch size')
parser.add_argument('--gamma', default = 0.99, help='discount factor')
parser.add_argument('--tau', default = 0.001, help='for soft update of target parameters')
parser.add_argument('--lr', default = 0.0001, help='learning rate')
parser.add_argument('--update_every', default = 5, help='how often to update the q-network')
parser.add_argument('--act_every', default = 1, help='how often to take new actions')
parser.add_argument('--fc1', default = 32, help='fc1 units')
parser.add_argument('--fc2', default = 64, help='fc2 units')
parser.add_argument('--BN', default = "True", help='flag for using batch norm')
parser.add_argument('--seed', default = 42, help='seed')
parser.add_argument('--algo', default = 'dqn', help='learning algorithm')
parser.add_argument('--eps_init', default = 1.0, help='initial epsilon value')
parser.add_argument('--eps_decay', default = 0.995, help='epsilon decay rate')
parser.add_argument('--train_episodes', default = 1000, help='Number of training episodes')
parser.add_argument('-t', '--test_name', help='Input test name is required', required=True)
```
3. After training, the results will be saved in "tests" folder

4. If you want to visualize your trained agent, just run in the terminal:
```
>> python navigate.py -t "mytest"
```

### Results

The best model found was a DQN agent with Batch Normalization, able to solve the environment in only 234 episodes! This was achieved after a careful hyperparameter tuning, leading to a significant improvement compared to the initial base case settings.

![Results for base case and best model settings](https://github.com/thenickben/drlnd-navigation/blob/master/figures/best_vs_basecase-plot.png)

#### Requirements

The python packages needed are shown in requirements.txt