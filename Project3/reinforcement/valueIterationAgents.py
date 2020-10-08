# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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

import mdp, util

from learningAgents import ValueEstimationAgent
import collections
import operator

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # print(self.values)
        # states = self.mdp.getStates()
        # print(states)

        numIterations = self.iterations
        states = self.mdp.getStates()
        # qvalues = util.Counter()

        for i in range(numIterations):
            # return Counter(dict.copy(self))
            qvalues = self.values.copy()
            for s in states:
                if not self.mdp.isTerminal(s):
                    temp_action = self.getAction(s)
                    temp_qvalue = self.getQValue(s, temp_action)
                    qvalues[s] = temp_qvalue
            self.values = qvalues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # returns the Q-value of the (state, action) pair given by the value function given by self.values.
        finalSum = 0
        next_transState = []
        next_transProb = []
        # next_transState, next_transProb = self.mdp.getTransitionStatesAndProbs(state,action)
        for s, tp in self.mdp.getTransitionStatesAndProbs(state, action):
            next_transState.append(s)
            next_transProb.append(tp)

        #iterate through states and transition probabilites
        for (s,tp) in zip(next_transState,next_transProb):
            #value iteration formula= transition probability * [(Reward) + (discount factor)(value)]
            temp_reward = self.mdp.getReward(state, action, s)
            temp_discount = self.discount
            temp_value = self.getValue(s)
            finalSum += (tp*(temp_reward+(temp_discount*temp_value)))

        return finalSum

        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # computes the best action according to the value function given by self.values.
        allActions = self.mdp.getPossibleActions(state)
        allResults = []

        if len(allActions)>0:
            for action in allActions:
                #call computeQValueFromValues to calculate q value iteration sum of all successor states possible for this possible action
                temp_result = self.computeQValueFromValues(state, action)
                allResults.append([temp_result, action])
            # print(max(allResults))
            return (max(allResults)[1])

        #if there are no legal actions (in terminal state), return None
        else:
            return None
            # print(allResults)

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    #Question 4
    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # Cyclic Value Iteration:
        # In the first iteration, only update the value of the first state in the states list.
        # In the second iteration, only update the value of the second.
        # Keep going until you have updated the value of each state once, then start back at the first state for the subsequent iteration.
        # If the state picked for updating is terminal, nothing happens in that iteration.
        numIterations = self.iterations
        states = self.mdp.getStates()
        # print(states)
        # print(numIterations)
        # qvalues = util.Counter()

        for i in range(numIterations):
            # print(i)
            # return Counter(dict.copy(self))
            qvalues = self.values.copy()
            if (i>len(states)-1):
                s = states[i%len(states)]
            else:
                s = states[i]
            # print(s)
            terminal = self.mdp.isTerminal(s)
            if not terminal:
                temp_action = self.getAction(s)
                temp_qvalue = self.getQValue(s, temp_action)
                qvalues[s] = temp_qvalue
            self.values = qvalues

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    #Question 5
    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()

        #compute predecessors of all states
        predecs = dict()
        for i in states:
            predecs[i] = []

        for state in states:
            terminal = self.mdp.isTerminal(state)
            if not terminal:
                allActions = self.mdp.getPossibleActions(state)
                for action in allActions:
                    next_transState = []
                    next_transProb = []
                    # next_transState, next_transProb = self.mdp.getTransitionStatesAndProbs(state,action)
                    for s, tp in self.mdp.getTransitionStatesAndProbs(state, action):
                        terminal = self.mdp.isTerminal(s)
                        if not terminal:
                            next_transState.append(s)
                            next_transProb.append(tp)

                    #iterate through states and transition probabilites
                    for (s,tp) in zip(next_transState,next_transProb):
                        terminal = self.mdp.isTerminal(s)
                        # if not terminal:
                            #value iteration formula= transition probability * [(Reward) + (discount factor)(value)]
                            # temp_reward = self.mdp.getReward(state, action, s)
                            # temp_discount = self.discount
                            # temp_value = self.getValue(s)
                            # finalSum += (tp*(temp_reward+(temp_discount*temp_value)))

                        if tp > 0:
                            predecs[s].append(state)

        #initialize an empty priorityQueue
        pQueue = util.PriorityQueue()

        # For each non-terminal state s, do:
            # (note: to make the autograder work for this question, you must iterate over states in the order returned by self.mdp.getStates())
        for state in states:
            terminal = self.mdp.isTerminal(state)
            if not terminal:
                # Find the absolute value of the difference between the current value of s in self.values
                # and the highest Q-value across all possible actions from s (this represents what the value should be); call this number diff.
                # Do NOT update self.values[s] in this step.
                allActions = self.mdp.getPossibleActions(state)
                if len(allActions)>0:
                    #temporary max q value assignment
                    max_qvalue = self.computeQValueFromValues(state, allActions[0])
                    for action in allActions:
                        temp_qvalue = self.computeQValueFromValues(state, allActions[0])
                        if (temp_qvalue > max_qvalue):
                            max_qvalue = temp_qvalue

                        diff = max_qvalue - self.values[s]
                        diff = abs(diff)

                        # Push s into the priority queue with priority -diff (note that this is negative).
                        # We use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
                        pQueue.push(state, -diff)


        # For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        numIterations = self.iterations
        for i in range(numIterations):
            # If the priority queue is empty, then terminate.
            if (pQueue.isEmpty()):
                break
            else:
                # Pop a state s off the priority queue.
                s = pQueue.pop()

                # Update s's value (if it is not a terminal state) in self.values.
                terminal =  self.mdp.isTerminal(s)
                if not terminal:
                    allActions = self.mdp.getPossibleActions(s)
                    if len(allActions)>0:
                        #temporary max qvalue assignment
                        max_qvalue = self.getQValue(s, allActions[0])
                        for action in allActions:
                            temp_qvalue = self.getQValue(s, action)
                            if (temp_qvalue > max_qvalue):
                                max_qvalue = temp_qvalue
                    self.values[s] = max_qvalue

                print('success')
                # For each predecessor p of s, do:
                for predec in predecs[s]:
                    # print('success')
                    #
                    # # Find the absolute value of the difference between the current value of p in self.values
                    # # and the highest Q-value across all possible actions from p (this represents what the value should be); call this number diff.
                    # # Do NOT update self.values[p] in this step.
                    allActions = self.mdp.getPossibleActions(predec)
                    # print(allActions)
                    for a in allActions:
                        max_qvalue = self.computeQValueFromValues(predec, allActions[0])
                        for action in allActions:
                            temp_qvalue = self.computeQValueFromValues(predec, action)
                            if (temp_qvalue > max_qvalue):
                                max_qvalue = temp_qvalue
                        diff = max_qvalue - self.values[predec]
                        diff = abs(diff)
                    #
                    # If diff > theta,
                    if (diff > self.theta):
                        # push p into the priority queue with priority -diff (note that this is negative),
                        # as long as it does not already exist in the priority queue with equal or lower priority.
                        # As before, we use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.

                        #built in update function for priorityQueue checks if the element exists in the queue already ( do not have to repeat)
                        pQueue.update(predec, -diff)
