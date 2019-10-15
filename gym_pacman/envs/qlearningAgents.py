# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from gym_pacman.envs.game import *
from gym_pacman.envs.featureExtractors import *
from gym_pacman.envs.learningAgents import ReinforcementAgent
import gym_pacman.envs.util as util
import numpy as np
import random, math


class QLearningAgent(ReinforcementAgent):

    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.q_values = {}

    def getQValue(self, state, action):
        if (state, action) in self.q_values:
            return self.q_values[state, action]
        else:
            return 0.0

    def setQValue(self, state, action, value):
        self.q_values[(state, action)] = value

    def computeValueFromQValues(self, state):
        q_values = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        if not len(q_values): return 0.0
        return max(q_values)

    def computeActionFromQValues(self, state):
        best_q_value = self.getValue(state)
        best_actions = {}

        for action in self.getLegalActions(state):
            if self.getQValue(state, action) == best_q_value:
                best_actions.add(action)

        if len(best_actions) == 0:
            return None
        else:
            return random.choice(best_actions)

    def getAction(self, state):
        # Pick Action
        legal_actions = self.getLegalActions(state)
        action = None

        if util.flipCoin(self.epsilon):
            action = random.choice(legal_actions)
        else:
            action = self.getPolicy(state)

        return action

    def update(self, state, action, next_state, reward):
        alpha = self.alpha
        q_value = self.getQValue(state, action)
        discount = self.discount
        next_value = self.getQValue(next_state)

        new_value = (1-alpha)*q_value + alpha((reward + discount * next_value))

        self.setQValue(state, action, new_value)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action