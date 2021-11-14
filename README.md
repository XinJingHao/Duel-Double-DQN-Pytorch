# DQN/DDQN-Pytorch
This is a **clean and robust Pytorch implementation of DQN and Double DQN**. Here is the training curve:  

<img src="https://github.com/XinJingHao/DQN-DDQN-Pytorch/blob/main/IMGs/DQN_DDQN_result.png"/>
All the experiments are trained with same hyperparameters.

A quick render here:

![avatar](https://github.com/XinJingHao/DQN-DDQN-Pytorch/blob/main/IMGs/Render%20of%20DDQN.gif)

## Dependencies
gym==0.18.3  
numpy==1.21.2  
pytorch==1.8.1  

## How to use my code
### Train from scratch
run **'python main.py'**, where the default enviroment is CartPole-v1.  
### Play with trained model
run **'python main.py --write False --render True --Loadmodel True --ModelIdex 50000'**  
### Change Enviroment
If you want to train on different enviroments, just run **'python main.py --EnvIdex 1'**.  
The --EnvIdex can be set to be 0 and 1, where   
'--EnvIdex 0' for 'CartPole-v1'  
'--EnvIdex 1' for 'LunarLander-v2'   
### Visualize the training curve
You can use the tensorboard to visualize the training curve. History training curve is saved at '\runs'
### Hyperparameter Setting
For more details of Hyperparameter Setting, please check 'main.py'
### References
DQN: Mnih V , Kavukcuoglu K , Silver D , et al. Playing Atari with Deep Reinforcement Learning[J]. Computer Science, 2013. 

Double DQN: Hasselt H V , Guez A , Silver D . Deep Reinforcement Learning with Double Q-learning[J]. Computer ence, 2015.
