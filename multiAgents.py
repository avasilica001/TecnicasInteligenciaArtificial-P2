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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

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
        """
        "*** YOUR CODE HERE ***"

        #El Pacman utilizará la función que maximiza (maxFunction)
        def maxFunction(gameState, depth):
            actualDepth = depth + 1
            if actualDepth == self.depth:
                return self.evaluationFunction(gameState)
            elif gameState.isWin() or gameState.isLose():   #Evaluo si se ganó o perdió el juego
                return self.evaluationFunction(gameState)
            maxValue = -100000000 # Inicializo con el peor caso que seria - infinito
            actions = gameState.getLegalActions(0)  #Obtengo los posibles movimientos en base al estado
            for a in actions:
                successor = gameState.generateSuccessor(0,a)
                maxValue = max(maxValue,minFunction(successor,actualDepth,1)) #Voy iterando recursivamente por todos los sucesores quedandome con el valor maximo
            return maxValue
        
        #El resto de los agentes son los fantasmas y utilizaran la función que minimiza (minFunction)
        def minFunction(gameState,depth, agentIndex):
            minValue = 100000000 #Inicializo con el peor caso que seria + infinito
            nGhosts = gameState.getNumAgents() - 1 # Cantidad de fantasmas, -1 ya que uno de los agentes es el Pacman con indice 0
            if gameState.isWin() or gameState.isLose():   #Evaluo si el juego terminó o no
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            for a in actions:
                successor= gameState.generateSuccessor(agentIndex,a)
                if agentIndex == (nGhosts):
                    minValue = min(minValue,maxFunction(successor,depth))
                else:
                    minValue = min(minValue,minFunction(successor,depth,agentIndex+1))
            return minValue
        
        actions = gameState.getLegalActions(0) #Posibles movimientos
        actualScore = -100000000 #Inicializo el estado del pacman como el peor caso posible (siendo - infinito)
        nextAction = '' #La acción final que tomara el pacman
        for a in actions:
            nextState = gameState.generateSuccessor(0,a) #En base a las acciones posibles, voy generando los estados sucesores
            #Considerando al pacman como el vaor root, llamo a minFunction ya que el siguiente nivel del árbol será el caso de un fantasma
            score = minFunction(nextState,0,1)
            if score > actualScore:  # Me quedo con la acción de valor que minimiza de todos los sucesores.
                nextAction = a
                actualScore = score
        return nextAction
        #util.raiseNotDefined()

       
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
                #Used only for pacman agent hence agentIndex is always 0.
        def maxLevel(gameState,depth,alpha, beta):
            currDepth = depth + 1
            if gameState.isWin() or gameState.isLose() or currDepth==self.depth:   #Terminal Test 
                return self.evaluationFunction(gameState)
            maxvalue = -999999
            actions = gameState.getLegalActions(0)
            alpha1 = alpha
            for action in actions:
                successor= gameState.generateSuccessor(0,action)
                maxvalue = max (maxvalue,minLevel(successor,currDepth,1,alpha1,beta))
                if maxvalue > beta:
                    return maxvalue
                alpha1 = max(alpha1,maxvalue)
            return maxvalue
        
        #For all ghosts.
        def minLevel(gameState,depth,agentIndex,alpha,beta):
            minvalue = 999999
            if gameState.isWin() or gameState.isLose():   #Terminal Test 
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(agentIndex)
            beta1 = beta
            for action in actions:
                successor= gameState.generateSuccessor(agentIndex,action)
                if agentIndex == (gameState.getNumAgents()-1):
                    minvalue = min (minvalue,maxLevel(successor,depth,alpha,beta1))
                    if minvalue < alpha:
                        return minvalue
                    beta1 = min(beta1,minvalue)
                else:
                    minvalue = min(minvalue,minLevel(successor,depth,agentIndex+1,alpha,beta1))
                    if minvalue < alpha:
                        return minvalue
                    beta1 = min(beta1,minvalue)
            return minvalue

        # Alpha-Beta Pruning
        actions = gameState.getLegalActions(0)
        currentScore = -999999
        returnAction = ''
        alpha = -999999
        beta = 999999
        for action in actions:
            nextState = gameState.generateSuccessor(0,action)
            # Next level is a min level. Hence calling min for successors of the root.
            score = minLevel(nextState,0,1,alpha,beta)
            # Choosing the action which is Maximum of the successors.
            if score > currentScore:
                returnAction = action
                currentScore = score
            # Updating alpha value at root.    
            if score > beta:
                return returnAction
            alpha = max(alpha,score)
        return returnAction

        #util.raiseNotDefined()

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
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
