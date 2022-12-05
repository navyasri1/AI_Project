# searchAgents.py
# ---------------
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


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState
    
    def getGoalState(self):
        return self.goal

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state, direction='Forward'):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position

    if "backward_search" in info:
        xy2 = problem.getStartState()
    else:
        xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def manhattanDistance(position, goal):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    if "backward_search" in info:
        xy2 = problem.getStartState()
    else:
        xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

def euclideanDistance(position, goal):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState, costFn = lambda x:1):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"
        self.costFn = costFn
        self.type="CP"

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        return [self.startingPosition,tuple([False] * len(self.corners))]
        util.raiseNotDefined()
        
    def getGoalState(self):
        # goal state is the farthest corner

        farthest_corner = None
        cur_max = -99999
        for corner in self.corners:
            distance_to_corner = manhattanDistance(self.startingPosition, corner)
            if cur_max < distance_to_corner:
                cur_max = distance_to_corner
                farthest_corner = corner
        
        return ((farthest_corner, tuple([True] * len(self.corners))))

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        if set(self.corners) == set(state[1]):
            return True

        return False
        # util.raiseNotDefined()

    def getSuccessors(self, state, direction='Forward'):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """
        successors = []
        if direction =='Forward':
            # if state[0] in self.corners: # CHECK
            #     state[1].add(state[0])

            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                # Add a successor state to the successor list if the action is legal
                # Here's a code snippet for figuring out whether a new position hits a wall:
                #   x,y = currentPosition
                #   dx, dy = Actions.directionToVector(action)
                #   nextx, nexty = int(x + dx), int(y + dy)
                #   hitsWall = self.walls[nextx][nexty]

                "*** YOUR CODE HERE ***"
                
                x,y = state[0]
                dx, dy = Actions.directionToVector(action)
                nextx, nexty = int(x + dx), int(y + dy)
                if not self.walls[nextx][nexty]:
                    tmp_visited_corners = list(tuple(state[1]))

                    if (nextx, nexty) in self.corners:
                        tmp_visited_corners[self.corners.index((nextx,nexty))] = True

                    nextState = [(nextx, nexty),tuple(tmp_visited_corners)]
                    # cost = self.costFn(nextState)
                    successors.append( [ nextState, action, 1] )
        else:

            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                # Add a successor state to the successor list if the action is legal
                # Here's a code snippet for figuring out whether a new position hits a wall:
                #   x,y = currentPosition
                #   dx, dy = Actions.directionToVector(action)
                #   nextx, nexty = int(x + dx), int(y + dy)
                #   hitsWall = self.walls[nextx][nexty]

                "*** YOUR CODE HERE ***"
                
                x,y = state[0]
                dx, dy = Actions.directionToVector(action)
                nextx, nexty = int(x + dx), int(y + dy)
                if not self.walls[nextx][nexty]:
                    tmp_visited_corners = list(tuple(state[1]))

                    if state[0] in self.corners and (nextx, nexty) not in self.corners:
                        tmp_visited_corners[self.corners.index(state[0])] = False

                    nextState = [(nextx, nexty),tuple(tmp_visited_corners)]
                    #cost = self.costFn(nextState)
                    successors.append( [ nextState, action, 1] )

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem, info={}):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    cur_location = state[0]
    heuristic_value = 0

    if 'backward_search' in info:
        visited_corners = list(state[1])
        other_corners = []

        temp = []
        for i in range(len(corners)):
            if visited_corners[i]:
                temp.append(corners[i])
        
        for corner_1 in temp:
            for corner_2 in temp:
                if corner_1 != corner_2:
                    other_corners.append(manhattanDistance(corner_1, corner_2))
        
        heuristic_value = min(other_corners) if len(other_corners) else 0

    else:
        # for i in range(len(heuristics)):
        #     heuristics[i] = manhattanDistance(cur_location, corners[i])
        #     heuristics[i] = (heuristics[i] * (4.0 - len(state[1])))

        yet_to_visit = list(set(corners) - set(state[1]))

        for i in range(len(yet_to_visit)):
            heuristic_value = max(heuristic_value, manhattanDistance(cur_location, yet_to_visit[i]))
            # cur_location, cur_dist = find_next_closest_point(cur_location, list(yet_to_visit))
            # heuristic_value += cur_dist
            # yet_to_visit -= {cur_location}


    return heuristic_value

    # print("heuristics", heuristics)
    # print("heuristics state", state)
    # print("heuristics", heuristics)
    # return min(heuristics) # Default to trivial solution

def find_next_closest_point(start, points):
    dists = [0] * len(points)
    
    for i in range(len(dists)):
        dists[i] = manhattanDistance(start, points[i])
    
    return (list(points)[dists.index(min(dists))], min(dists))

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        self.heuristicInfo['FoodSearchProblem'] = self.start[1].copy()
        return self.start
    
    def getGoalState(self):
        farthest_food = None
        locations_of_food = self.start[1].asList()

        if not len(locations_of_food):
            return self.start
        
        cur_max = -99999

        for food_pos in locations_of_food:
            distance_to_food = mazeDistance(self.start[0],food_pos, self.startingGameState)
            if cur_max < distance_to_food:
                cur_max = distance_to_food
                farthest_food = food_pos
        
        food_locations = self.start[1].copy()

        for i in range(food_locations.width):
            for j in range(food_locations.height):
                food_locations[i][j] = False

        return tuple((farthest_food, food_locations))

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state, direction='Forward'):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                if direction == 'Forward':
                    nextFood = state[1].copy()
                    nextFood[nextx][nexty] = False
                    self.heuristicInfo['FoodSearchProblem'] = nextFood
                    successors.append( ( ((nextx, nexty), nextFood), action, 1) )
                else:
                    nextFood = state[1].copy()
                    nextFood[nextx][nexty] = self.heuristicInfo['FoodSearchProblem'][nextx][nexty]
                    successors.append( ( ((nextx, nexty), nextFood), action, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

def foodSearchHeuristic(state, problem, info={}):
    # Heuristic for food search.

    cur_location = state[0]
    heuristic_value = 0

    if ('backward_search' not in info and len(state[1].asList()) != len(problem.heuristicInfo['FoodSearchProblem'].asList()) or (len(problem.heuristicInfo['FoodSearchProblem'].asList()))):
        gamestate = problem.startingGameState

        if 'backward_search' in info:
            food_locations = problem.heuristicInfo['FoodSearchProblem'].asList()
        else:
            food_locations = state[1].asList()

        for food_location in food_locations:
            heuristic_value = max(heuristic_value, mazeDistance(cur_location, food_location, gamestate))

    return heuristic_value

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
