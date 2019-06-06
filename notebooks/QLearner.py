"""Implement QLearner, a Reinforcement Learning class"""

import numpy as np
import random as rand
from copy import deepcopy
from collections import deque

class QLearner(object):

    def __init__(self, num_states=100, num_actions=4, alpha=0.2,
        gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False):
        """The constructor QLearner() reserves space for keeping track of Q[s, a] for 
        the number of states and actions. It initializes Q[] with all zeros.

        Parameters:
        num_states: int, the number of states to consider
        num_actions: int, the number of actions available
        alpha: float, the learning rate used in the update rule. 
               Should range between 0.0 and 1.0 with 0.2 as a typical value
        gamma: float, the discount rate used in the update rule. 
               Should range between 0.0 and 1.0 with 0.9 as a typical value.
        rar: float, random action rate. The probability of selecting a random action 
             at each step. Should range between 0.0 (no random actions) to 1.0 
             (always random action) with 0.5 as a typical value.
        radr: float, random action decay rate, after each update, rar = rar * radr. 
              Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
        dyna: int, conduct this number of dyna updates for each regular update. 
              When Dyna is used, 200 is a typical value.
        verbose: boolean, if True, your class is allowed to print debugging 
                 statements, if False, all printing is prohibited.
        """        
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.memory = deque(maxlen=2000)

        # Keep track of the latest state and action which are initialized to 0
        self.s = 0
        self.a = 0
        
        # Initialize a Q table which records and updates Q value for
        # each action in each state
        self.Q = np.zeros(shape=(num_states, num_actions))
        # Keep track of the number of transitions from s to s_prime for when taking 
        # an action a when doing Dyna-Q
        self.T = {}
        # Keep track of reward for each action in each state when doing Dyna-Q
        self.R = np.zeros(shape=(num_states, num_actions))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, s, r, done=False, update=True):
        if update:
            return self.query(s, r, done=done)
        else:
            return self.query_set_state(s)

    def query_set_state(self, s):
        """Find the next action to take in state s. Update the latest state and action 
        without updating the Q table. Two main uses for this method: 1) To set the  
        initial state, and 2) when using a learned policy, but not updating it.

        Parameters:
        s: The new state
        
        Returns: The selected action to take in s
        """
        if rand.uniform(0.0, 1.0) < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = self.Q[s, :].argmax()

        self.s = s
        self.a = action
        if self.verbose: 
            print ("s =", s,"a =",action)
        return action

    def query(self, s_prime, r, done=False):
        """Find the next action to take in state s_prime. Update the latest state 
        and action and the Q table. Update rule:
        Q'[s, a] = (1 - α) · Q[s, a] + α · (r + γ · Q[s', argmax a'(Q[s', a'])]).

        Parameters:
        s_prime: int, the new state
        r: float, a real valued immediate reward for taking the previous action
        
        Returns: The selected action to take in s_prime
        """
        self.remember(self.s, self.a, r, s_prime, done)
        # Update the Q value of the latest state and action based on s_prime and r
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] \
                                    + self.alpha * (r + self.gamma 
                                    * self.Q[s_prime, self.Q[s_prime, :].argmax()])

        # Implement Dyna-Q
        if self.dyna > 0:
            # Update the reward table
            self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] \
                                        + self.alpha * r
            
            if (self.s, self.a) in self.T:
                if s_prime in self.T[(self.s, self.a)]:
                    self.T[(self.s, self.a)][s_prime] += 1
                else:
                    self.T[(self.s, self.a)][s_prime] = 1
            else:
                self.T[(self.s, self.a)] = {s_prime: 1}
            
            Q = deepcopy(self.Q)
            for i in range(self.dyna):
                s = rand.randint(0, self.num_states - 1)
                a = rand.randint(0, self.num_actions - 1)
                if (s, a) in self.T:
                    # Find the most common s_prime as a result of taking a in s
                    s_pr = max(self.T[(s, a)], key=lambda k: self.T[(s, a)][k])
                    # Update the temporary Q table
                    Q[s, a] = (1 - self.alpha) * Q[s, a] \
                                + self.alpha * (self.R[s, a] + self.gamma 
                                * Q[s_pr, Q[s_pr, :].argmax()])
            # Update the Q table of the learner once Dyna-Q is complete
            self.Q = deepcopy(Q)
        
        # Find the next action to take and update the latest state and action
        a_prime = self.query_set_state(s_prime)
        self.rar *= self.radr
        if self.verbose: 
            print ("s =", s_prime,"a =",a_prime,"r =",r)
        return a_prime

    def replay(self, batch_size=32):
        return None
