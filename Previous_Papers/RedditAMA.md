q: different algo compared to traditional rl algos
a: yes, because of 1.) 2+ player game 2.) imperfect info -> no knowledge of all the states
   no real textbook on IIG (imperfect information games) -> read papers

review articles + keynote videos:
* Keynote “New Results for Solving Imperfect-Information Games” at the Association for the Advancement of Artificial Intelligence Annual Conference (AAAI), 2019, available on Vimeo. (https://vimeo.com/313942390)
* Keynote “Super-Human AI for Strategic Reasoning: Beating Top Pros in Heads-Up No-Limit Texas Hold’em” at the International Joint Conference on Artificial Intelligence (IJCAI), available on YouTube. (https://www.youtube.com/watch?v=xrWulRY_t1o)
* Solving Imperfect-Information Games. (http://www.cs.cmu.edu/~sandholm/Solving%20games.Science-2015.pdf) Science 347(6218), 122-123, 2015.
* Abstraction for Solving Large Incomplete-Information Games. (http://www.cs.cmu.edu/~sandholm/game%20abstraction.aaai15SMT.pdf) In AAAI, Senior Member Track, 2015.
* The State of Solving Large Incomplete-Information Games, and Application to Poker. (http://www.cs.cmu.edu/~sandholm/solving%20games.aimag11.pdf) AI Magazine, special issue on Algorithmic Game Theory, Winter, 13-32, 2010.

q: Pluribus focuses on being minimally-exploitable rather than maximally-exploiting
   Sota in expliting strategy?
a: Safe Opponent Exploitation (http://www.cs.cmu.edu/~sandholm/safeExploitation.teac15.pdf).
   Online Convex Optimization for Sequential Decision Processes and Extensive-Form Games (http://www.cs.cmu.edu/~gfarina/2018/laminar-regret-aaai19/).

q: why was stack size reset, does it perform worse if it's not?
a: varying stack size makes harder to evaluate the bot

q: Is Pluribus actually able to adapt to the way its opponents are playing, or does it learn purely from playing against itself? 
a: Pluribus does not adapt to the way its opponents play.
   It treated each hand that it played against the humans individually and did not carry over knowledge from one hand to another.

q: lecture or video on this topic?
a: libratus - https://www.youtube.com/watch?v=2dX0lwaQRX0

q: does it work similarly to alpha go(montecarlo tree search)?
a: Monte Carlo Tree Search as used in AlphaGo does not work in imperfect-information games.

q: opponent may play according to blueprint, but can also fold. What if there is only one action (e.g., can only play a card). 
a: The key idea here is that Pluribus understands the players are not limited to a single strategy beyond the leaf nodes, but rather can choose among multiple strategies for the remainder of the game. Those strategies could be anything, and there are many different ways those strategies can be determined. In a game like Uno for example, you could have the different strategies be playing different cards.

q: different parameters make different strategies?
a: probably, previously there was a poker ai that had several versions and swapped inbetween those.

a: this paper is mentioned quite a bit: Online Convex Optimization for Sequential Decision Processes and Extensive-Form Games (http://www.cs.cmu.edu/~gfarina/2018/laminar-regret-aaai19/).

q: what made it possible to reduce the required computational resources?
a: The big breakthrough was the depth-limited search algorithm. This allowed us to shift a lot of the load from the blueprint computation to the online search algorithm, and the online search algorithm is relatively much more efficient

a: "Pluribus does not adapt its strategy to the observed tendencies of its opponents."

a: I think opponent adaptation/exploitation is still a very interesting AI challenge. 
   I do think that top pros could beat weak players by more than Pluribus would (though I do think Pluribus would still make a ton of money off of weak players). 
   The current state of the art for opponent adaptation is pretty disappointing. For example, in the days of the Annual Computer Poker Competition, the bots that won the opponent exploitation category didn’t do any adaptation, they would just play an approximate Nash equilibrium! 
   But it’s clear you can do really well in poker without opponent adaptation, so I think it might be better to look at other domains where opponent adaptation is necessary to do well.

q: things you tried but didn't work?
a: One thing in particular we tried was "safe" search techniques (see this paper for details on what that means).
   Pluribus we use a technique that's sort of half-way between safe and unsafe search. Unsafe search is theoretically dangerous and could potentially lead to really bad strategies. Safe search fixes this in theory, but we found it was much more expensive to run. 
   Seems that safe search is not necessary for 6-player poker. 

a: In particular, our depth-limited solving paper led to a huge reduction in the computational cost of generating strong poker AI bots.
   Those breakthroughs are the reason we can now make a superhuman six-player no-limit Texas hold’em bot with the equivalent of less than $150 worth of compute.


  
