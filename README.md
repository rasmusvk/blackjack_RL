# blackjack_RL
Teaching an agent blackjack using reinforcement learning for the course AE4350 Bio-inspired Intelligence and learning for Aerospace Applications at TU Delft.

This is the main code used for the assignment. It is quite easy to change it to account for more or less decks, train it for different hyperparameters, 
evaluate the perfect strategy, change the policy to a random one etc.

To evaluate the perfect strategy, you need to change the __init__ of the Q-values to 1 for hits and -1 for stands, and run it with a greedy policy.
To change the policy to a random one, you need to change the __init__ of the Q-values to random values between -1 and 1, and run it with a greedy policy.
