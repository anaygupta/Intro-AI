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

#--------------------------------------------------------------------------------
# Anay Gupta
# CSE 471 Intro to AI
# Rao - T/Th 1:30 pm
# Project 3

# Extra Online sources used:
# https://stackoverflow.com/questions/1663807/how-to-iterate-through-two-lists-in-parallel
# https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
# https://www.geeksforgeeks.org/copy-python-deep-copy-shallow-copy/
# https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/mdps-exact-methods.pdf
# https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/node19.html
# https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html
# https://www.freecodecamp.org/news/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe/

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math, numpy as np

# Learning by trial and error
class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"

        self.QVALUES = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.QVALUES[(state, action)]
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        # print(self.QVALUES)
        # qvalues = util.Counter()
        allActions = self.getLegalActions(state)

        #assign max to something temporarily
        max_qvalue = self.getQValue(state, allActions[0])
        if len(allActions) > 0:
            for action in allActions:
                temp_qvalue = self.getQValue(state, action)
                if (temp_qvalue > max_qvalue):
                    max_qvalue = temp_qvalue
            return max_qvalue
        else:
            return 0.0

        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #break ties randomly

        allActions = self.getLegalActions(state)
        actionList = []
        qValues = util.Counter()
        #assign max to something temporarily
        max_qvalue = self.getQValue(state, allActions[0])
        if len(allActions) > 0:
            for action in allActions:
                temp_qvalue = self.getQValue(state, action)
                if (temp_qvalue > max_qvalue):
                    max_qvalue = temp_qvalue
        if len(allActions) > 0:
            for action in allActions:
                temp_qvalue = self.getQValue(state, action)

                if(max_qvalue == temp_qvalue):
                    actionList.append(action)
            return random.choice(actionList)
                # return max_qvalue
        else:
            return 0.0

        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"

        # You can choose an element from a list uniformly at random by calling the random.choice function.
        # You can simulate a binary variable with probability p of success by using util.flipCoin(p),
        #     which returns True with probability p and False with probability 1-p.

        bestAction = self.computeActionFromQValues(state)

        #if 1-epsilon, get bestAction
        if(util.flipCoin(1-self.epsilon)):
            return bestAction
        else:
            return random.choice(legalActions)

        # util.raiseNotDefined()
        # return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        # you only access Q values by calling getQValue
        qvalues = util.Counter()
        allActions = self.getLegalActions(nextState)
        #
        # if len(allActions) > 0:
        for a in allActions:
        #         # print(i)
        #         # return Counter(dict.copy(self))
            temp_qvalue = self.getQValue(nextState, a)
            qvalues[a] = temp_qvalue
        # max_index = qvalues.index(max(qvalues))
        qiteration = (1-self.alpha)* (self.QVALUES[(state, action)]) + self.alpha * ((reward) + (self.discount)*(qvalues[qvalues.argMax()]))
        self.QVALUES[(state, action)] = qiteration

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
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
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.QVALUES = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        return self.QVALUES[(state, action)]

        # util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        # you only access Q values by calling getQValue
        qvalues = util.Counter()
        allActions = self.getLegalActions(nextState)
        #
        # if len(allActions) > 0:
        for a in allActions:
        #         # print(i)
        #         # return Counter(dict.copy(self))
            temp_qvalue = self.getQValue(nextState, a)
            qvalues[a] = temp_qvalue
        # max_index = qvalues.index(max(qvalues))
        qiteration = (1-self.alpha)* (self.QVALUES[(state, action)]) + self.alpha * ((reward) + (self.discount)*(qvalues[qvalues.argMax()]))
        self.QVALUES[(state, action)] = qiteration

        # util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
