# poker-bot

## Overview
This repository contains the project for the Deep learning class (course code: VITMAV45) at the Budapest University of Technology and Economics. Our project focuses on reinforcement learning with the aim of training an agent in a poker environment.

## Environment
RLcard is an easy-to-use toolkit (http://rlcard.org/overview.html) that provides Leduc Hold’em environment which is a smaller version of Limit Texas Hold’em.  This version of poker was introduced in the research paper *Bayes’ Bluff: Opponent Modeling in Poker* in 2012 (https://arxiv.org/abs/1207.1411). 
- 6 cards: two pairs of King, Queen and Jack
- 2 players
- 2 rounds
- Raise amounts of 2 in the first round and 4 in the second round
- 2-bet maximum
- 0-14 chips for the agent and for the opponent

First round: players put 1 unit in the pot and are dealt 1 card, then start betting. <br />
Second round: 1 public card is revealed, then the players bet again. <br />
End: the player wins, whose hand has the same rank as the public card or has higher rank than the opponent. 

#### State Representation
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

#### Actions
There are 4 action types which are encoded as below.
| State |      Meaning      |  
|----------|:-------------:|
| 0 |  Call | 
| 1 |    Raise |  
| 2 | Fold |
| 3 |  Check | 

#### Payoff
The reward is based on big blinds per hand.
| Reward |      Meaning      |  
|----------|:-------------:|
| R |  the player wins R times of the amount of big blind | 
| -R | the player loses R times of the amount of big blind |  
