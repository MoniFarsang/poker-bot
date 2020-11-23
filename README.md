# poker-bot

## Overview
This repository contains the project for the Deep learning class (course code: VITMAV45) at the Budapest University of Technology and Economics. Our project focuses on reinforcement learning with the aim of training an agent in a poker environment.

## Code
The [presented code for the first milestone](https://github.com/MoniFarsang/poker-bot/blob/main/training_agent/train.py) is based on the RLcard github repository [example code](https://github.com/datamllab/rlcard/blob/master/examples/leduc_holdem_cfr.py). It is used as a presentation that the chosen environment works and the agent is ready to train.</br >
</br >
The [code for the second milestone](https://github.com/MoniFarsang/poker-bot/tree/main/agent) is a DQN agent in PyTorch. We used the RLcard [DQN agent](https://github.com/datamllab/rlcard/blob/master/rlcard/agents/dqn_agent.py) written in TensorFlow as a base and created a more powerful, more manageable, and easy to use code in Pytorch. This implementation is an advanced Q-learning agent in two aspects. First, it uses a replay buffer to store past experiences and we can sample training data from it periodically.  Second, to make the training more stable, another Q-network is used as a target network in order to backpropagate through it and train the policy Q-network. These features are described in the Nature paper [*Human-level control through deep reinforcement learning*](https://www.nature.com/articles/nature14236).</br >
Furthermore, as an extra component, we added the opportunity of a more aggressive playing strategy. In case of the given action has the maximum q-value, the agent chooses the Raise action instead if it is a valid action. The possible settings are displayed below: </br>
| Strategy settings |      Meaning      |  
|----------|:-------------:|
| 0 |  Using action with maximum value (default in DQN) | 
| 1 |  If action *Call* has the maximum value, we use *Raise* action if possible |  
| 2 |  If action *Check* has the maximum value, we use *Raise* action if possible |
| 3 |  If action *Fold* has the maximum value, we use *Raise* action if possible |

The agent can be trained and evaluated against a [random agent](https://github.com/datamllab/rlcard/blob/master/rlcard/agents/random_agent.py) and a [pre-trained agent](https://github.com/datamllab/rlcard/blob/master/rlcard/models/pretrained_models.py). 
| Opponent settings |      Meaning      |  
|----------|:-------------:|
| 0 |  Random agent | 
| 1 |  Pre-trained NFSP agent |  

These can be set in the [training code](https://github.com/MoniFarsang/poker-bot/blob/main/training_dqn.py) for the DQN agent. </br >

### References:</br >
These references were used during the implementation of the DQN agent in PyTorch. </br >
https://github.com/datamllab/rlcard/blob/master/rlcard/agents/dqn_agent.py </br >
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html </br >
https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c

### Using Dockerfile 
The Dockerfile contains the list of system dependencies. After building the image, which gives a simple containerization of our application, the training runs successfully in its container. </br >  </br >
For building the image use following command:  </br >
`$ docker build --tag IMAGE_NAME:TAG .`  </br >
e.g. `$ docker build --tag poker-bot:1.0 .` </br >  </br >
For running the image:  </br >
`$ docker run IMAGE_NAME:TAG`  </br >
e.g. `$ docker run poker-bot:1.0` </br >

### Using Notebook format
#### First milestone
A notebook version is presented in the repository as well. If you want to get a quick look at our first milestone results, we recommend to choose [this one](https://github.com/MoniFarsang/poker-bot/blob/main/poker_bot_notebook.ipynb). 
#### Second milestone
For the second milestone, we present two versions, one in the [Leduc Hold'em](https://github.com/MoniFarsang/poker-bot/blob/main/poker-bot-dqn-leduc-notebook.ipynb) and the other in the [Limit Hold'em](https://github.com/MoniFarsang/poker-bot/blob/main/poker-bot-dqn-limit-notebook.ipynb) environment. In the Leduc Hold'em environment we can play agent the pre-trained agent.

## Environment
[RLcard](http://rlcard.org/overview.html) is an easy-to-use toolkit that provides [Leduc Hold’em environment](http://rlcard.org/games.html#leduc-hold-em) which is a smaller version of Limit Texas Hold’em.  This version of poker was introduced in the research paper [Bayes’ Bluff: Opponent Modeling in Poker](https://arxiv.org/abs/1207.1411) in 2012. 
### Limited environment
- 6 cards: two pairs of King, Queen and Jack
- 2 players
- 2 rounds
- Raise amounts of 2 in the first round and 4 in the second round
- 2-bet maximum
- 0-14 chips for the agent and for the opponent

First round: players put 1 unit in the pot and are dealt 1 card, then start betting. <br />
Second round: 1 public card is revealed, then the players bet again. <br />
End: the player wins, whose hand has the same rank as the public card or has higher rank than the opponent. 

### State Representation
The state is encoded as a vector of length 36, the indices and their meaning is presented below.

| Index |      Meaning      |  
|----------|:-------------:|
| 0 |  Jack in hand | 
| 1 |    Queen in hand  |  
| 2 | King in hand |
| 3 |  Jack as public card | 
| 4 |    Queen as public card   |  
| 5 | King as public card |
| 6-20 |  0-14 chips for the agent | 
| 21-35 |    0-14 chips for the opponent  |  

### Actions
There are 4 action types which are encoded as below.
| Action |      Meaning      |  
|----------|:-------------:|
| 0 |  Call | 
| 1 |    Raise |  
| 2 | Fold |
| 3 |  Check | 

### Payoff
The reward is based on big blinds per hand.
| Reward |      Meaning      |  
|----------|:-------------:|
| R |  the player wins R times of the amount of big blind | 
| -R | the player loses R times of the amount of big blind |  
