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
        score, action =  self.minimaxFunction(gameState, True, '', 0, 0)
        return action

        util.raiseNotDefined()

    def minimaxFunction(self, gameState, maxBool, agentAction, currDepth, curr_agentIndex):

        #max agent (pacman) playing
        if (maxBool):
            pacmanActions = gameState.getLegalActions(curr_agentIndex)
            if ((currDepth == self.depth) or (not pacmanActions)):
                #self.evaluationFunction defaults to gameState
                return [self.evaluationFunction(gameState), agentAction]
            else:
                scoreAction_arr = []
                for action in pacmanActions:
                    pacman_successorState = gameState.generateSuccessor(curr_agentIndex, action)

                    curr_ghostIndex = 1
                    temp = self.minimaxFunction(pacman_successorState, False, action, currDepth, curr_ghostIndex)
                    scoreAction_arr.append(temp)
                scoreAction_arr.sort()
                scoreAction_arr = sorted(scoreAction_arr, reverse = True)

                # print('This is the score Action array:')
                # print(scoreAction_arr)

            if len(scoreAction_arr) > 0:
                if currDepth == 0:
                    return scoreAction_arr[0]
                    #return max score and action pair for root node
                else:
                    return [scoreAction_arr[0][0], agentAction]
                    #return max score and preceding action
            else:
                return self.evaluationFunction(gameState, agentAction)


        else:
            ghostActions = gameState.getLegalActions(curr_agentIndex)
            if ((currDepth == self.depth) or (not ghostActions)):
                #self.evaluationFunction defaults to gameState
                return [self.evaluationFunction(gameState), agentAction]
            else:
                numGhosts = gameState.getNumAgents() - 1
                    # List to store the score of each ghost action
                scoreAction_arr = []
                for action in ghostActions:
                    ghost_successorState = gameState.generateSuccessor(curr_agentIndex, action)

                    if curr_agentIndex == numGhosts:
                        temp = self.minimaxFunction(ghost_successorState, True, action, currDepth+1, 0)
                        scoreAction_arr.append(temp)

                    else:
                        # As this is not the last ghost, pass the ghost successor state to the NEXT ghost
                        # and append the result to scoreAction_arr
                        temp = self.minimaxFunction(ghost_successorState, False, action, currDepth, curr_agentIndex + 1)
                        # calling min of next ghost
                        scoreAction_arr.append(temp)

                scoreAction_arr = sorted(scoreAction_arr)

            # If scoreAction_arr is not empty, return the node with minimum return value, with the action of the previous node
            if len(scoreAction_arr) > 0:
                return [scoreAction_arr[0][0], agentAction]
            # If scoreAction_arr is empty, return the evaluated value of the node, with the action of the previous node
            else:
                return [self.evaluationFunction(gameState), agentAction]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabetaFunction(self, gameState, action, currDepth, curr_agentIndex, maxBool, alpha, beta):
            # alpha: the alpha value propagated from the parent
            # beta: the beta value propagated from the parent
        # This function returns [score, action, resultingAlpha, resultingBeta], where:
            # score is the best possible score
            # action is the corresponding move
            # resultingAlpha is the evaluated alpha value of the current node
            # resultingBeta is the evaluated beta value of the current node


        INF = float("INF")

        #max node
        if (maxBool):

            # If maximum depth is reached, return the evaluated value of the node as alpha
            if (currDepth >= self.depth) or (gameState.isWin()) or (gameState.isLose()):
                alpha = self.evaluationFunction(gameState)
                temp = self.evaluationFunction(gameState)
                result = [temp, action, alpha, beta]
                return result

            # List to store the score of each Pacman action
            scoreAction_arr = []

            # Get legal Pacman actions and generate Pacman successor states
            for pacmanAction in gameState.getLegalActions(curr_agentIndex):
                #pruning - if beta becomes less than alpha, stop exploring further successors
                if beta < alpha:
                    return [score, resultingAction, alpha, beta]

                pacmanSuccessorState = gameState.generateSuccessor(curr_agentIndex, pacmanAction)
                currentGhostIndex = 1
                score, resultingAction, resultingAlpha, resultingBeta = self.alphabetaFunction(pacmanSuccessorState, pacmanAction, currDepth, currentGhostIndex, False, alpha, beta)


                if resultingBeta > alpha:
                    alpha = resultingBeta

                scoreAction_arr.append([score, resultingAction])

            scoreAction_arr = sorted(scoreAction_arr, reverse=True)

            if len(scoreAction_arr) > 0:

                if currDepth != 0:
                    return [scoreAction_arr[0][0], action, scoreAction_arr[0][0], beta]
                else:
                    return [scoreAction_arr[0][0], scoreAction_arr[0][1], scoreAction_arr[0][0], beta]

            else:
                score, resultingAction, resultingAlpha, resultingBeta = self.alphabetaFunction(gameState, action, currDepth + 1, 1, False, alpha, beta)
                alpha = resultingBeta
                return [score, resultingAction, alpha, beta]





        #min node
        else:
            if currDepth >= self.depth or gameState.isWin() or gameState.isLose():
                beta = self.evaluationFunction(gameState)
                return [self.evaluationFunction(gameState), action, alpha, beta]
            numGhosts = gameState.getNumAgents() - 1
            scoreAction_arr = []
            # Get legal ghost actions and generate ghost successor states
            for ghostAction in gameState.getLegalActions(curr_agentIndex):
                # Pruning - if beta becomes less than alpha, stop exploring further successors
                if beta < alpha:
                    return [score, resultingAction, alpha, beta]
                ghost_successorState = gameState.generateSuccessor(curr_agentIndex, ghostAction)


                if curr_agentIndex == numGhosts:

                    score, resultingAction, resultingAlpha, resultingBeta = self.alphabetaFunction(ghost_successorState, ghostAction, currDepth + 1, 0, True, alpha, beta) # call max with depth + 1
                    if resultingAlpha < beta:
                        beta = resultingAlpha
                else:

                    score, resultingAction, resultingAlpha, resultingBeta = self.alphabetaFunction(ghost_successorState, ghostAction, currDepth, curr_agentIndex + 1, False, alpha, beta) # calling min of next ghost
                    if beta > resultingBeta:
                        beta = resultingBeta
                scoreAction_arr.append([score, resultingAction])
            scoreAction_arr = sorted(scoreAction_arr)
            if len(scoreAction_arr) > 0:
                return [scoreAction_arr[0][0], action, alpha, scoreAction_arr[0][0]]
            else:
                score, resultingAction, resultingAlpha, resultingBeta = self.alphabetaFunction(gameState, action, currDepth + 1, 0, True, alpha, beta)
                beta = resultingAlpha
                return[score, resultingAction, alpha, beta]

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import sys
        INF = float("inf")

        score, action, alpha, beta = self.alphabetaFunction(gameState, "dummy", 0, 0, True, -INF, INF)

        return action
        # util.raiseNotDefined()

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
    import sys
        MAX = sys.maxint
        MIN = -MAX

        # No Food left, return MAX
        if len(newFood) == 0:
            return MAX

        newPacmanPosition = newPos
        newGhostPositions = successorGameState.getGhostPositions()
        currentFood = currentGameState.getFood().asList()

        # To store manhattan distances between pacman's new position and ghost's new position
        nonscaryGhostDistances = []
        scaryGhostDistances = []

        # To store the new positions of ghosts
        nonscaryGhosts = []
        scaryGhosts = []

        # Calculate the manhattan distances between Pacman's new position and each new ghost position
        for i in range(0, len(newGhostPositions)):
            if newScaredTimes[i] > 0:
                nonscaryGhostDistances.append(manhattanDistance(newPacmanPosition, newGhostPositions[i]))
                nonscaryGhosts.append(newGhostPositions[i])
            else:
                scaryGhostDistances.append(manhattanDistance(newPacmanPosition, newGhostPositions[i]))
                scaryGhosts.append(newGhostPositions[i])

        # To store the manhattan distances between Pacman's new position and new food positions
        foodDistances = []
        for elem in newFood:
            foodDistances.append(manhattanDistance(newPacmanPosition, elem))


        closestFoodDistance = MAX
        closestScaryGhostDist = MAX
        closestNonScaryGhostDist = MAX

        # Find the manhattan distance to the closest non scary ghost
        if len(nonscaryGhostDistances) > 0:
            nonscaryGhostDistances = sorted(nonscaryGhostDistances)
            closestNonScaryGhostDist = nonscaryGhostDistances[0]

        # Find the manhattan distance to the closest scary ghost
        if len(scaryGhostDistances) > 0:
            scaryGhostDistances = sorted(scaryGhostDistances)
            closestScaryGhostDist = scaryGhostDistances[0]

        # Sort manhattan distances from Pacman's new position to the new Food position in ascending order and find the closest food distance
        foodDistances = sorted(foodDistances)
        closestFoodDistance = foodDistances[0]

        # Threshold value is used to check whether the ghost is within a certain range of Pacman
        threshold = 4

        # If a scary ghost is within the range of Pacman
        if closestScaryGhostDist < threshold:
            # If Pacman is losing in the next move, then return MIN
            if newPacmanPosition in scaryGhosts:
                return MIN
            # If Pacman is not losing in next move, then return the closest manhattan distance between the ghost and the Pacman
            else:
                return closestScaryGhostDist

        # If a non scary ghost is within the range of Pacman
        if closestNonScaryGhostDist < threshold:
            # If the manhattan distance between Pacman and closest non scary ghost is greater than the manhattan distance from closest food,
            # then chase food
            # else chase the non scary ghost
            if closestNonScaryGhostDist > closestFoodDistance:
                return threshold + 1 - closestFoodDistance
            else:
                return threshold + 1 - closestNonScaryGhostDist

        # If Pacman eats the food in the next move, then return MAX value so that Pacman eats it
        if newPacmanPosition in currentFood:
            return MAX

        # If Pacman eats the pellet in the next move, then return MAX value so that the Pacman eats it
        if newPos in successorGameState.getCapsules():
            return MAX

        # If no ghost is in range and no capsule is nearby, chase the closest food
        return threshold + 1 - closestFoodDistance
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
