## Special environment
This library is for experimentation.

Mainly for simulating tournament, and comparing it to other agents. 
Currently only working in leduc-holdem, and probably due to computational limits we won't be using limit-holdem (for this specific environment). 

Most notable changes are connected to the states, where we use relative chip measurement, meaning that the agent will compare itself to the other player's chips. Negative if the
agent has less chips, positive otherwise. 

### References
This part originates from the [RLCard repository](https://github.com/datamllab/rlcard/tree/master/rlcard).

