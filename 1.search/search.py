# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    closed = set()
    paths = {}
    from util import Stack
    fringe = Stack()
    fringe.push(problem.getStartState())
    paths[problem.getStartState()] = []
    while fringe.isEmpty() != 1:
        front = fringe.pop()
        if problem.isGoalState(front):
            goal = front
            break
        if front not in closed:
            closed.add(front)
            for child in problem.getSuccessors(front):
                fringe.push(child[0])
                paths[child[0]] = paths[front][:]
                paths[child[0]].append(child[1])
    return paths.get(goal, [])


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    explored = set()
    paths = {}
    from util import Queue
    fringe = Queue()
    fringe.push(problem.getStartState())
    fringe_nodes = set()
    fringe_nodes.add(problem.getStartState())
    paths[problem.getStartState()] = []
    while fringe.isEmpty() != 1:
        front = fringe.pop()
        fringe_nodes.remove(front)
        if problem.isGoalState(front):
            goal = front
            break
        explored.add(front)
        for child in problem.getSuccessors(front):
            if child[0] not in fringe_nodes and child[0] not in explored:
                fringe.push(child[0])
                fringe_nodes.add(child[0])
                paths[child[0]] = paths[front][:]
                paths[child[0]].append(child[1])
    return paths.get(goal, [])


def uniformCostSearch(problem):
    explored = set()
    paths = {problem.getStartState(): []}
    cost = {problem.getStartState(): 0}
    fringe_nodes = set()
    from util import PriorityQueue
    fringe = PriorityQueue()
    fringe.push(problem.getStartState(), 0)
    fringe_nodes.add(problem.getStartState())
    while fringe.isEmpty() != 1:
        front = fringe.pop()
        fringe_nodes.remove(front)
        if problem.isGoalState(front):
            goal = front
            break
        if front not in explored:
            explored.add(front)
            for child in problem.getSuccessors(front):
                if child[0] not in explored and child[0] not in fringe_nodes:
                    fringe.push(child[0], cost[front] + child[2])
                    fringe_nodes.add(child[0])
                    paths[child[0]] = paths[front][:]
                    paths[child[0]].append(child[1])
                    cost[child[0]] = cost[front] + child[2]
                elif child[0] in fringe_nodes and cost[front] + child[2] < cost[child[0]]:
                    cost[child[0]] = cost[front] + child[2]
                    fringe.update(child[0], cost[front] + child[2])
                    paths[child[0]] = paths[front][:]
                    paths[child[0]].append(child[1])
    return paths.get(goal, [])


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    explored = set()
    paths = {problem.getStartState(): []}
    cost = {problem.getStartState(): heuristic(problem.getStartState(), problem)}
    fringe_nodes = set()
    from util import PriorityQueue
    fringe = PriorityQueue()
    fringe.push(problem.getStartState(), heuristic(problem.getStartState(), problem))
    fringe_nodes.add(problem.getStartState())

    while fringe.isEmpty() != 1:
        front = fringe.pop()
        fringe_nodes.remove(front)
        if problem.isGoalState(front):
            goal = front
            break
        if front not in explored:
            explored.add(front)
            for child in problem.getSuccessors(front):
                if child[0] not in explored and child[0] not in fringe_nodes:
                    fringe.push(child[0], cost[front] + child[2] + heuristic(child[0], problem))
                    fringe_nodes.add(child[0])
                    paths[child[0]] = paths[front][:]
                    paths[child[0]].append(child[1])
                    cost[child[0]] = cost[front] + child[2]
                elif child[0] in fringe_nodes and cost[front] + child[2] < cost[child[0]]:
                    cost[child[0]] = cost[front] + child[2]
                    fringe.update(child[0], cost[front] + child[2] + heuristic(child[0], problem))
                    paths[child[0]] = paths[front][:]
                    paths[child[0]].append(child[1])
    return paths.get(goal, [])


def reverse_path(path):
    reverse_list = list()
    l = {'East':'West', 'West':'East', 'North': 'South', 'South':'North'}
    for p in path:
        reverse_list.append(l[p])

    return reverse_list[::-1]


def path_update(problem, fringe1, fringe2, closed1, fringe_o_nodes1, fringe_o_nodes2, cost1, cost2, paths1, itr1, itr2):
    fringe1.push(itr1, 2 * cost2[itr1])
    fringe_o_nodes1.add(itr1)
    closed1.add(itr2)
    for child in problem.getSuccessors(itr2):
        if (child[0] in fringe_o_nodes2 or child[0] in closed1) and (
                cost1[child[0]] <= cost1[itr2] + child[2]):
            continue
        if child[0] in fringe_o_nodes2 or child[0] in closed1:
            if child[0] in fringe_o_nodes2:
                # fringe2.remove_by_value(child[0])
                l = [i[2][0] for i in fringe2.heap]
                del fringe2.heap[l.index(child[0])]
                fringe_o_nodes2.remove(child[0])
            if child[0] in closed1:
                closed1.remove(child[0])

        cost1[child[0]] = cost1[itr2] + child[2]
        fringe2.push(child[0], 2 * cost1[child[0]])
        fringe_o_nodes2.add(child[0])
        paths1[child[0]] = paths1[itr2][:]
        paths1[child[0]].append(child[1])


def MM0Search(problem):
    gf_start = 0
    gb_goal = 0
    # U = float('inf')

    from util import PriorityQueue

    fringe_openf = PriorityQueue()
    fringe_openf.push(problem.getStartState(), 2 * gf_start)
    fringe_openf_nodes = set()
    closed_f = set()
    paths_f = {problem.getStartState(): []}
    cost_f = {problem.getStartState(): gf_start}

    fringe_openb = PriorityQueue()
    fringe_openb.push(problem.goal, 2 * gb_goal)
    fringe_openb_nodes = set()
    closed_b = set()
    paths_b = {problem.goal: []}
    cost_b = {problem.goal: gb_goal}

    fringe_openf_nodes.add(problem.getStartState())
    fringe_openb_nodes.add(problem.goal)

    while fringe_openf.isEmpty() != 1 and fringe_openb.isEmpty() != 1:
        front = fringe_openf.pop()
        back = fringe_openb.pop()
        fringe_openf_nodes.remove(front)
        fringe_openb_nodes.remove(back)
        C = min(cost_f[front], cost_b[back])

        if front == back:
            return paths_f[front] + reverse_path(paths_b[back])
        if front in closed_b:
            return paths_f[front] + reverse_path(paths_b[front])
        if back in closed_f:
            return paths_f[back] + reverse_path(paths_b[back])

        if C == cost_f[front]:
            path_update(problem, fringe_openb, fringe_openf, closed_f, fringe_openb_nodes, fringe_openf_nodes, cost_f,
                        cost_b, paths_f, back, front)

        elif C == cost_b[back]:
            path_update(problem, fringe_openf, fringe_openb, closed_b, fringe_openf_nodes, fringe_openb_nodes, cost_b,
                        cost_f, paths_b, front, back)

    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
mm0 = MM0Search