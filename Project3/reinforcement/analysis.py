# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9    #default discount of 0.9
    # answerNoise = 0.2       #default noise of 0.2
    answerNoise = 0.0

    return answerDiscount, answerNoise

# Prefer the close exit (+1)
# Risking the cliff (-10)
def question3a():
    answerDiscount = 0.1
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# Prefer the close exit (+1)
# Avoiding the cliff (-10)
def question3b():
    answerDiscount = 0.3
    answerNoise = 0.3
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# Prefer the distant exit (+10)
# Risking the cliff (-10)
def question3c():
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# Prefer the distant exit (+10)
# Avoiding the cliff (-10)
def question3d():
    answerDiscount = 0.9
    answerNoise = 0.3
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# Avoid both exits
def question3e():
    answerDiscount = 1.0
    answerNoise = 0.0
    answerLivingReward = 100
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = None
    answerLearningRate = None
    # return answerEpsilon, answerLearningRate
    # doesnt seem like there is an epsilon or learning rate that will learn optimal policy after 50 iterations
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
