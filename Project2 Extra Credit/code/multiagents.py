# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import sys
# import pdb
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        # scores = []
        # for action in legalMoves:
        #     scores.append(self.evaluationFunction(gameState, action))
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        # print('new iteration')
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        # print(successorGameState)   #prints a grid of the next state with pacman being G
        newPos = successorGameState.getPacmanPosition()
        # print(newPos)    #get the new position of pacman after action is implemented
        newFood = successorGameState.getFood()
        # print(newFood[0])
        newGhostStates = successorGameState.getGhostStates()
        # print(newGhostStates)
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # print(newScaredTimes[0])

        #Evalution Function for pacman:
            # Select actions on the basis of the current percept
            # Ignoring the rest of the percept
        "*** YOUR CODE HERE ***"
        #if successor pacman state is different from new ghost states

        #Ghost Location
        #see if new positions distance will be further away from ghost's current position
        currentPos = currentGameState.getPacmanPosition()
        ghostPosition = currentGameState.getGhostPosition(1)

        score = successorGameState.getScore()
        olddist_toGhost = util.manhattanDistance(currentPos, ghostPosition)
        newdist_toGhost = util.manhattanDistance(newPos, ghostPosition)
        if(newdist_toGhost > olddist_toGhost):
            score += 50

        #Food Location
        #check if less number of food particles

        currFood = currentGameState.getFood()
        currFood = currFood.asList()
        newFood = newFood.asList()

        #asList turns these into a list of coordinates
        if (len(newFood) < len(currFood)):
            score += 50
        # print(len(currFood.asList()))
        # print(currFood.asList())
        # print(len(newFood.asList()))
        # print(newFood.asList())

        #find the closest food particle in the current state
        #find the closest food particle in the next state
        #if distance to closest food particle in next state is closer add 5

        if (len(currFood)>1):
            currFoodDistance = util.manhattanDistance(currentPos, currFood[0])
            for i in range(1, len(currFood)):
                temp_currFoodDistance = util.manhattanDistance(currentPos, currFood[i])
                if (temp_currFoodDistance < currFoodDistance):
                    currFoodDistance = temp_currFoodDistance
        elif len(currFood) >0:
            currFoodDistance = util.manhattanDistance(currentPos, currFood[0])
            # print(currFood)
            # print(action)
        else:
            currFoodDistance = None

        if (len(newFood)>1):
            newFoodDistance = util.manhattanDistance(newPos, newFood[0])
            for i in range(1, len(newFood)):
                temp_newFoodDistance = util.manhattanDistance(newPos, newFood[i])
                if (temp_newFoodDistance < newFoodDistance):
                    newFoodDistance = temp_newFoodDistance
        elif len(newFood) > 0:
            newFoodDistance = util.manhattanDistance(newPos, newFood[0])
            # print(newFood)
            # print(action)
        else:
            newFoodDistance = None

        if (newFoodDistance is not None) and (currFoodDistance is not None) and (newFoodDistance < currFoodDistance):
            score += 50


        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        """
        - Minimax Agent should work with any number of ghosts
        - Minimax Tree will have multiple min layers (one for each ghost) for every max layer
        - Code should also expand game tree to arbitrary depth
        - Score leaves of minimax tree with supplied self.evaluationFunction
        - MinimaxAgent extends MultiAgentSearchAgent giving it access to:
            - self.depth
            - self.evaluationFunction
        - depth 2 search involves Pacman and each ghost moving two times
        """


        # pdb.set_trace()
        # print(gameState)
        # print(gameState.getLegalActions(0))
        #agent 1 is pacman (0)
        minimax =  self.minimaxFunction(gameState, True, '', 0, 0)
        score = minimax[0]
        action = minimax[1]

        return action
        # util.raiseNotDefined()

    def minimaxFunction(self, gameState, maxBool, agentAction, currDepth, curr_agentIndex):

        #max agent (pacman) playing
        if (maxBool):
            actions = gameState.getLegalActions(curr_agentIndex)
            if ((currDepth == self.depth) or (not actions)):
                #self.evaluationFunction defaults to scoreEvaluationFunction
                temp_score = self.evaluationFunction(gameState)
                return [temp_score, agentAction]
            else:
                scoreAction_arr = []
                for action in actions:
                    pacman_successorState = gameState.generateSuccessor(curr_agentIndex, action)
                    curr_ghostIndex = 1
                    temp = self.minimaxFunction(pacman_successorState, False, action, currDepth, curr_ghostIndex)
                    scoreAction_arr.append(temp)

                # scoreAction_arr.sort()
                #sorted ascending
                scoreAction_arr = sorted(scoreAction_arr)

                # print('This is the score Action array:')
                # print(scoreAction_arr)

            if len(scoreAction_arr) > 0:
                if currDepth == 0:
                    return scoreAction_arr[len(scoreAction_arr)-1]
                    #return max score and action pair for root node
                else:
                    scoreAction = [scoreAction_arr[len(scoreAction_arr)-1][0], agentAction]
                    return scoreAction
                    #return max score and preceding action
            else:
                scoreAction = self.evaluationFunction(gameState, agentAction)
                return scoreAction


        #min agent (ghost) playing
        else:
            ghostActions = gameState.getLegalActions(curr_agentIndex)
            if ((currDepth == self.depth) or (not ghostActions)):
                #self.evaluationFunction defaults to scoreEvaluationFunction
                return [self.evaluationFunction(gameState), agentAction]
            else:
                numAgents = gameState.getNumAgents() - 1
                scoreAction_arr = []

                for action in ghostActions:
                    ghost_successorState = gameState.generateSuccessor(curr_agentIndex, action)

                    #if last min agent (last ghost)
                    if curr_agentIndex == numAgents:
                        #set depth to max agent (pacman)
                        newDepth = currDepth + 1
                        temp = self.minimaxFunction(ghost_successorState, True, action, newDepth, 0)
                        scoreAction_arr.append(temp)

                    #ghosts remaining
                    else:
                        #iterate to next min agent (next ghost)
                        new_agentIndex = curr_agentIndex + 1
                        temp = self.minimaxFunction(ghost_successorState, False, action, currDepth, new_agentIndex)
                        scoreAction_arr.append(temp)

                #sorted ascending
                scoreAction_arr = sorted(scoreAction_arr)

            if len(scoreAction_arr) > 0:
                scoreAction = [scoreAction_arr[0][0], agentAction]
                return scoreAction
                #return min score and action pair for root node
            else:
                scoreAction = [self.evaluationFunction(gameState), agentAction]
                return scoreAction
                #return min score and preceding action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        temp_beta = float("inf")
        temp_alpha = -float("inf")

        alphabeta = self.alphabetaFunction(gameState, True, '', 0, 0, temp_alpha, temp_beta)
        score = alphabeta[0]
        action = alphabeta[1]
        alpha = alphabeta[2]
        beta = alphabeta[3]

        return action
        # util.raiseNotDefined()

    def alphabetaFunction(self, gameState, maxBool, agentAction, currDepth, curr_agentIndex, alpha, beta):

        infinity = float("INF")

        #max agent (pacman) playing
        if (maxBool):

            #check for max depth
            if (currDepth >= self.depth) or (gameState.isWin()) or (gameState.isLose()):
                alpha = self.evaluationFunction(gameState)
                temp = self.evaluationFunction(gameState)
                scoreAction = [temp, agentAction, alpha, beta]
                return scoreAction

            else:
                scoreAction_arr = []
                actions = gameState.getLegalActions(curr_agentIndex)

                for action in actions:
                    if beta < alpha:
                        scoreAction = [score, resultAction, alpha, beta]
                        return scoreAction
                        #best, no more exploration

                    pacman_successorState = gameState.generateSuccessor(curr_agentIndex, action)
                    curr_ghostIndex = 1
                    score, resultAction, resultAlpha, resultBeta = self.alphabetaFunction(pacman_successorState, False, action, currDepth, curr_ghostIndex, alpha, beta)

                    if resultBeta > alpha:
                        alpha = resultBeta

                    temp_element = [score, resultAction]
                    scoreAction_arr.append(temp_element)

                #sorted ascending
                scoreAction_arr = sorted(scoreAction_arr)

            if len(scoreAction_arr) > 0:
                #if root node
                if currDepth == 0:
                    scoreAction = [scoreAction_arr[len(scoreAction_arr)-1][0], scoreAction_arr[len(scoreAction_arr)-1][1], scoreAction_arr[len(scoreAction_arr)-1][0], beta]
                    return scoreAction
                    #return max score and action pair for root node, and alpha score
                else:
                    scoreAction = [scoreAction_arr[len(scoreAction_arr)-1][0], agentAction, scoreAction_arr[len(scoreAction_arr)-1][0], beta]
                    return scoreAction
                    #return max score and preceding action, alpha score
            else:
                score, resultAction, resultAlpha, resultBeta = self.alphabetaFunction(gameState, False, agentAction, currDepth + 1, 1, alpha, beta)

                alpha = resultBeta
                scoreAction = [score, resultAction, alpha, beta]
                return scoreAction


        #min agent (ghost) playing
        else:
            if (currDepth >= self.depth) or (gameState.isWin()) or (gameState.isLose()):
                beta = self.evaluationFunction(gameState)
                temp = self.evaluationFunction(gameState)
                scoreAction = [temp, agentAction, alpha, beta]
                return scoreAction

            numAgents = gameState.getNumAgents() - 1
            scoreAction_arr = []
            actions = gameState.getLegalActions(curr_agentIndex)

            for action in actions:
                if beta < alpha:
                    scoreAction = [score, resultAction, alpha, beta]
                    return scoreAction
                    #best, no more exploration


                ghost_successorState = gameState.generateSuccessor(curr_agentIndex, action)

                #if last min agent (last ghost)
                if curr_agentIndex == numAgents:
                    #set depth to max agent (pacman)
                    newDepth = currDepth+1
                    score, resultAction, resultAlpha, resultBeta = self.alphabetaFunction(ghost_successorState, True, action, newDepth, 0, alpha, beta)

                    if resultAlpha < beta:
                        beta = resultAlpha

                #ghosts remaining
                else:
                    #iterate to next min agent (next ghost)
                    new_agentIndex = curr_agentIndex+1
                    score, resultAction, resultAlpha, resultBeta = self.alphabetaFunction(ghost_successorState, False, action, currDepth, new_agentIndex, alpha, beta) # calling min of next ghost

                    if beta > resultBeta:
                        beta = resultBeta

                temp_element = [score, resultAction]
                scoreAction_arr.append(temp_element)

            #sorted ascending
            scoreAction_arr = sorted(scoreAction_arr)

            if len(scoreAction_arr) > 0:
                scoreAction = [scoreAction_arr[0][0], agentAction, alpha, scoreAction_arr[0][0]]
                return scoreAction
                #return min score and preceding action, beta score

            else:
                newDepth = currDepth+1
                score, resultAction, resultAlpha, resultBeta = self.alphabetaFunction(gameState, True, agentAction, newDepth, 0, alpha, beta)
                beta = resultAlpha
                scoreAction = [score, resultAction, alpha, beta]
                return scoreAction
                #return calculated value and preceding action, beta score

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"


    #Q1 Useful information you can extract from a GameState (pacman.py)
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    # newPos = successorGameState.getPacmanPosition()
    # newFood = successorGameState.getFood()
    # newGhostStates = successorGameState.getGhostStates()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    #
    # #Evalution Function for pacman:
    #     # Select actions on the basis of the current percept
    #     # Ignoring the rest of the percept
    # #if successor pacman state is different from new ghost states
    #
    # #Ghost Location
    # #see if new positions distance will be further away from ghost's current position
    # currentPos = currentGameState.getPacmanPosition()
    # ghostPosition = currentGameState.getGhostPosition(1)
    #
    # score = successorGameState.getScore()
    # olddist_toGhost = util.manhattanDistance(currentPos, ghostPosition)
    # newdist_toGhost = util.manhattanDistance(newPos, ghostPosition)
    #
    # #Food Location
    # #check if less number of food particles
    # currFood = currentGameState.getFood()
    # currFood = currFood.asList()
    #
    # if (len(newFood) < len(currFood)):
    #     score += 50

    score = scoreEvaluationFunction(currentGameState)


    currFood = currentGameState.getFood()
    currFood = currFood.asList()
    currentPos = currentGameState.getPacmanPosition()

    foodDistances = []
    for food in currFood:
         distance = -1*manhattanDistance(currentPos, food)
         foodDistances.append(distance)
    if not foodDistances:
        foodDistances.append(0)

    #add distance of the closest food particle to score
    return score + max(foodDistances)-


    # util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
