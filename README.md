# poker-bot

## Overview
This repository contains the project for the Deep learning class (course code: VITMAV45) at the Budapest University of Technology and Economics. Our project focuses on reinforcement learning with the aim of training an agent in a poker environment. After training, we can play against our pre-trained agent.

## Code
### First milestone
The [presented code for the first milestone](https://github.com/MoniFarsang/poker-bot/blob/main/training_agent/train.py) is based on the RLcard github repository [example code](https://github.com/datamllab/rlcard/blob/master/examples/leduc_holdem_cfr.py). It is used as a presentation that the chosen environment works and the agent is ready to train.</br >
### Second milestone
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

#### References:</br >
These references were used during the implementation of the DQN agent in PyTorch. </br >
https://github.com/datamllab/rlcard/blob/master/rlcard/agents/dqn_agent.py </br >
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html </br >
https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c

### Final code
In the final code, we saved the best agents after hyperparameter optimization. These pre-trained agents can be set as opponents in the Leduc and Limit Hold'em environment. The [playing game code](https://github.com/MoniFarsang/poker-bot/blob/main/game.py) runs in the Leduc Hold'em environment by default. You can choose between running code using Docker or using Colab notebooks. More details are written below.

### Using Dockerfile 
The Dockerfile contains the list of system dependencies. After building the image, which gives a simple containerization of our application, the game runs successfully in its container. </br >  </br >
For building the image use following command:  </br >
`$ docker build --tag IMAGE_NAME:TAG .`  </br >
e.g. `$ docker build --tag poker-bot:1.0 .` </br >  </br >
For running the image:  </br >
#### Playing in the Leduc Hold'em environment:
`$ docker run -ti IMAGE_NAME:TAG`
or
`$ docker run -ti IMAGE_NAME:TAG --env leduc`  </br >
e.g. `$ docker run -ti poker-bot:1.0`
or 
`$ docker run -ti poker-bot:1.0 --env leduc`
#### Playing in the Limit Hold'em environment:
`$ docker run -ti IMAGE_NAME:TAG --env limit`  </br >
e.g. `$ docker run -ti poker-bot:1.0 --env limit` </br >

### Using Notebook format
#### First milestone
A notebook version is presented in the repository as well. If you want to get a quick look at our first milestone results, we recommend to choose [this one](https://github.com/MoniFarsang/poker-bot/blob/main/notebooks/first%20milestone/poker_bot_notebook.ipynb). 
#### Second milestone
For the second milestone, we present two versions, one in the [Leduc Hold'em](https://github.com/MoniFarsang/poker-bot/blob/main/notebooks/training%20DQN%20agent/poker-bot-dqn-leduc-notebook.ipynb) and the other in the [Limit Hold'em](https://github.com/MoniFarsang/poker-bot/blob/main/notebooks/training%20DQN%20agent/poker-bot-dqn-limit-notebook.ipynb) environment. After training the DQN agent in the Leduc Hold'em environment, you can play against it.
#### Final code
Our final code is presented in notebook format as well. You can play game against our pre-trained agents in the [Leduc Hold'em](https://github.com/MoniFarsang/poker-bot/blob/main/notebooks/playing%20game/poker-bot-dqn-leduc-game.ipynb) and [Limit Hold'em](https://github.com/MoniFarsang/poker-bot/blob/main/notebooks/playing%20game/poker-bot-dqn-limit-game.ipynb) environments.

## Environment
[RLcard](http://rlcard.org/overview.html) is an easy-to-use toolkit that provides [Limit Hold’em environment](http://rlcard.org/games.html#limit-texas-hold-em) and [Leduc Hold’em environment](http://rlcard.org/games.html#leduc-hold-em). The latter is a smaller version of Limit Texas Hold’em and it was introduced in the research paper [Bayes’ Bluff: Opponent Modeling in Poker](https://arxiv.org/abs/1207.1411) in 2012. 

### Limit Hold'em
- 52 cards
- Each player has 2 hole cards (face-down cards)
- 5 community cards (3 phases: flop, turn, river)
- 4 betting rounds
- Each player has 4 Raise actions in each round

#### State Representation in Limit Hold'em
The state is encoded as a vector of length 72. It can be splitted into two parts, the first part is the known cards (hole cards plus the known community cards). The second part is the number of Raise actions in the rounds. The indices and their meaning are presented below.

| Index |      Meaning      |  
|----------|:-------------:|
| 0-12 |  Spade A - Spade K | 
| 13-25 |    Heart A - Heart K  |  
| 26-38 | Diamond A - Diamond K |
| 39-51 |  Club A - Club K | 
| 52-56 |    Raise number in round 1   |  
| 57-61 | Raise number in round 2 |
| 62-66 |  Raise number in round 3 | 
| 67-71 |    Raise number in round 4  |  


### Leduc Hold'em
- 6 cards: two pairs of King, Queen and Jack
- 2 players
- 2 rounds
- Raise amounts of 2 in the first round and 4 in the second round
- 2-bet maximum
- 0-14 chips for the agent and for the opponent

First round: players put 1 unit in the pot and are dealt 1 card, then start betting. <br />
Second round: 1 public card is revealed, then the players bet again. <br />
End: the player wins, whose hand has the same rank as the public card or has higher rank than the opponent. 

#### State Representation in Leduc Hold'em
The state representation is different from the Limit Hold'em environment, its length is 36. The indices and their meaning are presented below.

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
Actions are the same in the Limit and the Leduc Hold'em environment. There are 4 action types which are encoded as below.
| Action |      Meaning      |  
|----------|:-------------:|
| 0 |  Call | 
| 1 |    Raise |  
| 2 | Fold |
| 3 |  Check | 

### Payoff
Payoff is the same in the Limit and the Leduc Hold'em environment. The reward is based on big blinds per hand.
| Reward |      Meaning      |  
|----------|:-------------:|
| R |  the player wins R times of the amount of big blind | 
| -R | the player loses R times of the amount of big blind | 

 
