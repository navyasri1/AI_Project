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
import math

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


def nullHeuristic(state, problem=None, info={}):
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


class mm_state:
    def __init__(self, n, action_sequence, cost_n=0, h_n=0, pr_n=0):
        self.n = n
        self.action_sequence = action_sequence
        self.cost_n = cost_n
        self.h_n = h_n
        self.pr_n = pr_n

        self.f_n = self.cost_n + self.h_n

    def getF(self):
        return self.f_n

    def getG(self):
        return self.cost_n

    def getPriority(self):
        return self.pr_n

    def getPosition(self):
        return self.n[0]

    def getNode(self):
        return self.n

    def getActionSequence(self):
        return self.action_sequence

def biDirectionalMeetInTheMiddle(problem, heuristic=nullHeuristic):
    forward_open_fringe = util.PriorityQueue()
    backward_open_fringe = util.PriorityQueue()

    forward_f = util.PriorityQueue()
    backward_f = util.PriorityQueue()

    forward_g = util.PriorityQueue()
    backward_g = util.PriorityQueue()

    forward_closed_list = dict()
    backward_closed_list = dict()

    # Helper variables
    forward_map = dict()
    backward_map = dict()

    final_action_sequence = list()

    # Problem

    start_state = tuple(problem.getStartState())
    goal_state = tuple(problem.getGoalState())

    start_n = mm_state(start_state, list(), h_n=heuristic(start_state, problem))
    goal_n = mm_state(goal_state, list(), h_n=heuristic(goal_state, problem, info={"backward_search": True}))

    # Bootstrapping initial values

    forward_f.push(start_n, start_n.getF())
    backward_f.push(goal_n, goal_n.getF())

    forward_g.push(start_n, start_n.getG())
    backward_g.push(goal_n, goal_n.getG())

    forward_open_fringe.push(start_n, start_n.getPriority())
    forward_map[start_state] = start_n

    backward_open_fringe.push(goal_n, goal_n.getPriority())
    backward_map[goal_state] = goal_n

    U = math.inf

    eps = 1  # For the pacman domain the smallest cost is believed to be 1

    while (not forward_open_fringe.isEmpty() and not backward_open_fringe.isEmpty()):
        C = min(forward_open_fringe.prmin(), backward_open_fringe.prmin())

        if U <= max(C, forward_open_fringe.prmin_node().getF(), backward_open_fringe.prmin_node().getF(),
                    forward_g.prmin_node().getG() + backward_g.prmin_node().getG() + eps):
            # return U

            print("Nodes generated : {}".format(
                len(forward_map) + len(backward_map) + len(forward_closed_list) + len(backward_closed_list) + 1))
            print("Cost: {}".format(U))

            return final_action_sequence

        elif U <= C:  # Theorem 10 in Holte et al. 2015

            print("Nodes generated : {}".format(
                len(forward_map) + len(backward_map) + len(forward_closed_list) + len(backward_closed_list) + 1))
            print("Cost: {}".format(U))

            return final_action_sequence

        if C == forward_open_fringe.prmin():
            # choose n which belongs to the open forward fringe whose priority is equal to prmin_f
            n = forward_open_fringe.pop()

            # move n from open fringe to closed fringe
            forward_f.remove(n)
            forward_g.remove(n)

            if n.getNode() in forward_map:
                forward_map.pop(n.n)

            forward_closed_list[n.n] = n

            # get every child c of n
            successors = problem.getSuccessors(n.n)

            for c in successors:
                child_state, child_action, child_cost = c
                child_state = tuple(child_state)
                child_node = None

                if child_state in forward_map:  # Check if c is present in Forward U Backward maps
                    child_node = forward_map[child_state]
                elif child_state in forward_closed_list:
                    child_node = forward_closed_list[child_state]

                if child_node != None:  # c is present in Forward U Backward maps
                    if child_node.getG() <= n.getG() + child_cost:
                        continue

                    # remove child from open and closed fringes
                    forward_open_fringe.remove(child_node)
                    if child_state in forward_map:
                        forward_map.pop(child_state)
                    if child_state in forward_closed_list:
                        forward_closed_list.pop(child_state)

                    forward_f.remove(child_node)
                    backward_f.remove(child_node)

                    # Recalculate g_f(c)
                    child_node.cost_n = (n.getG() + child_cost)
                    child_node.f_n = child_node.getG() + heuristic(child_node.n, problem)
                    child_node.action_sequence = n.getActionSequence() + [child_action]

                else:  # c is not present in Forward U Backward maps
                    child_node_action_sequence = n.getActionSequence() + [child_action]
                    child_node_cost = n.getG() + child_cost
                    child_node_heuristic = heuristic(child_state, problem)
                    child_node_pr = child_node_cost + max(child_node_heuristic, child_node_cost)

                    child_node = mm_state(child_state, child_node_action_sequence, child_node_cost,
                                          child_node_heuristic, child_node_pr)

                forward_open_fringe.push(child_node, child_node.getPriority())
                forward_map[child_state] = child_node
                forward_f.push(child_node, child_node.getF())
                forward_g.push(child_node, child_node.getG())

                # Line 17 of pseudocode. Check if child node is already reached from complementary side
                if child_state in backward_map:
                    U = min(U, child_node.getG() + backward_map[child_state].getG())

                    backward_segment = backward_map[child_state].getActionSequence()
                    backward_segment.reverse()

                    back_to_forward_key = {'North': 'South', 'South': 'North', 'East': 'West', 'West': 'East'}

                    for i in range(len(backward_segment)):
                        backward_segment[i] = back_to_forward_key[backward_segment[i]]

                    final_action_sequence = child_node.getActionSequence() + backward_segment

                    print("Path Found : " + str(final_action_sequence))
                    print("Forward cost \t: " + str(child_node.getG()))
                    print("Backward cost \t: " + str(backward_map[child_state].getG()))
                    print("Total cost \t: " + str(child_node.getG() + backward_map[child_state].getG()))

        else:  # Backward Search
            # choose n which belongs to the open backward fringe whose priority is equal to prmin_f
            n = backward_open_fringe.pop()

            # move n from open fringe to closed fringe
            backward_f.remove(n)
            backward_g.remove(n)

            if n.getNode() in backward_map:
                backward_map.pop(n.n)

            backward_closed_list[n.n] = n

            # get every child c of n
            successors = problem.getSuccessors(n.n, direction='Backward')

            for c in successors:
                child_state, child_action, child_cost = c
                child_state = tuple(child_state)
                child_node = None

                if child_state in backward_map:  # Check if c is present in Forward U Backward maps
                    child_node = backward_map[child_state]
                elif child_state in backward_closed_list:
                    child_node = backward_closed_list[child_state]

                if child_node != None:  # c is present in Forward U Backward maps
                    if child_node.getG() <= n.getG() + child_cost:
                        continue

                    # remove child from open and closed fringes
                    backward_open_fringe.remove(child_node)
                    if child_state in backward_map:
                        backward_map.pop(child_state)
                    if child_state in backward_closed_list:
                        backward_closed_list.pop(child_state)

                    forward_f.remove(child_node)
                    backward_f.remove(child_node)

                    # Recalculate g_f(c)
                    child_node.cost_n = (n.getG() + child_cost)
                    child_node.f_n = child_node.getG() + heuristic(child_node.n, problem, {'backward_search': True})
                    child_node.action_sequence = n.getActionSequence() + [child_action]

                else:  # c is not present in Forward U Backward maps
                    child_node_action_sequence = n.getActionSequence() + [child_action]
                    child_node_cost = n.getG() + child_cost
                    child_node_heuristic = heuristic(child_state, problem, {'backward_search': True})
                    child_node_pr = child_node_cost + max(child_node_heuristic, child_node_cost)

                    child_node = mm_state(child_state, child_node_action_sequence, child_node_cost,
                                          child_node_heuristic, child_node_pr)

                backward_open_fringe.push(child_node, child_node.getPriority())
                backward_map[child_state] = child_node
                backward_f.push(child_node, child_node.getF())
                backward_g.push(child_node, child_node.getG())

                # Line 17 of pseudocode. Check if child node is already reached from complementary side
                if child_state in forward_map:
                    U = min(U, child_node.getG() + forward_map[child_state].getG())

                    forward_segment = forward_map[child_state].getActionSequence()

                    backward_segment = child_node.getActionSequence()
                    backward_segment.reverse()

                    back_to_forward_key = {'North': 'South', 'South': 'North', 'East': 'West', 'West': 'East'}

                    for i in range(len(backward_segment)):
                        backward_segment[i] = back_to_forward_key[backward_segment[i]]

                    final_action_sequence = forward_segment + backward_segment

                    print("Path Found : " + str(final_action_sequence))
                    print("Forward cost \t: " + str(forward_map[child_state].getG()))
                    print("Backward cost \t: " + str(child_node.getG()))
                    print("Total cost \t: " + str(forward_map[child_state].getG() + child_node.getG()))

                    # util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
#mm0 = bidirectional_MM0
#mm = bidirectional_MM
mm = biDirectionalMeetInTheMiddle
mm0 = biDirectionalMeetInTheMiddle
