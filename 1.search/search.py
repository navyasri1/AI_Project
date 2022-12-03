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


def bidirectional_MM(problem, heuristic=nullHeuristic):
    """
    This function gets the minimum f value in the open_queue and g_value
    data structures. It goes through every single node in open_queue and
    then calculates the f_value of the node by adding the heuristic with
    the g_value of the node. That f value then gets added into a list
    where the list later returns the smallest f value from that list.
    """

    def min_f_value(open_queue, g_value):
        # create a min variable and set it to None
        min_f_val = None

        # go through every node in the open queue
        for node in open_queue:
            # get the state node
            node_state = node[2][0]

            # get the g value of node
            node_g_value = grab_the_path(node_state, g_value)

            # calculate the f value
            f_value = ((heuristic(node_state, problem)) + (node_g_value))

            # check if the min variable is empty
            if (min_f_val == None):
                # set min variable to f value
                min_f_val = f_value
            # if the min variable is not empty
            else:
                # store the smallest value
                min_f_val = min(min_f_val, f_value)

        # return the smallest f value
        return min_f_val

    """
    This function gets the minimum g value in the open_queue and g_value
    data structures. It goes through every single node in open_queue and
    then grabs the g_value of that node. That g value then gets stored
    into a list where that list later returns the smallest g value.
    """

    def min_g_value(open_queue, g_value):
        # create a min variable and set it to None
        min_g_val = None

        # go through every node in the open queue
        for node in open_queue:
            # get the state node
            node_state = node[2][0]

            # add the g value to the list
            g_val = grab_the_path(node_state, g_value)

            # check if the min variable is empty
            if (min_g_val == None):
                # set min variable to g value
                min_g_val = g_val
            # if the min variable is not empty
            else:
                # store the smallest value
                min_g_val = min(min_g_val, g_val)

        # return the smallest g value
        return min_g_val

    """
    This function finds the node where its priority value is equal to the
    minimum priority and its g value is equal to the minimum g value. Once
    those conditions are met, we will extract the state, path, and cost of
    the node and return it while removing the node from the open queue.
    """

    def find_node(open_queue, g_value, minimum_priority, minimum_g):
        # initialize the state, path, and cost variables to None
        node_state = None
        node_path = None
        node_cost = None

        # loop through every node in the list
        for node in open_queue:
            # get the state node
            state_node = node[2][0]

            # get the g value of node
            node_g_value = grab_the_path(state_node, g_value)

            # get the f value of the node
            f_value = ((heuristic(state_node, problem)) + (node_g_value))

            # find the priority value of the node
            priority_value = min(f_value, ((2) * (node_g_value)))

            # check if the priority value and g value are the minimum values
            if ((priority_value == minimum_priority) and (node_g_value == minimum_g)):
                # store the node state
                node_state = node[2][0]

                # store the node path
                node_path = node[2][1]

                # store the node cost
                node_cost = node[2][2]

                # remove the node from the open queue
                state_to_remove(open_queue, node_state)

                # break out of the for loop
                break

        # return the state, path, and cost
        return node_state, node_path, node_cost

    """
    This function checks if the state that we want to look for in the open_queue
    exists. If it does, then we return True. If it doesn't then we return False.
    """

    def state_in_list(open_queue, looking_for_state):
        # loop through the priority queue that's been turned into a heap
        for node in open_queue:
            # grab the state we are looking for in the node
            node_state = node[2][0]

            # check if the state node is the state we are looking for
            if (node_state == looking_for_state):
                # return true
                return True

        # return false
        return False

    """
    This function removes the state we want to get rid of from the open_queue. We
    add every single element in the open_queue into a list, grab the index from
    the list based on the state_we_want_to_remove, and then remove whatever was at
    that index. Then, we update the open_queue by storing the new queue that we
    were changing.
    """

    def state_to_remove(open_queue, state_we_want_to_remove):
        # copy the priority queue heap
        copy_heap = open_queue

        # create an empty list
        list_of_node_states = []

        # loop through the priority queue that's been turned into a heap
        for node in open_queue:
            # grab the state we are looking for in the node
            node_state = node[2][0]

            # add the node state into the list
            list_of_node_states.extend([node_state])

        # grab the index of the state we want to remove from the list
        state_index = list_of_node_states.index(state_we_want_to_remove)

        # delete whatever was at the state index
        del copy_heap[state_index]

        # update the priority queue heap by storing the manipulated copy into it
        open_queue = copy_heap

        # go back to algorithm
        return

    """
    This function finds out if the state we are looking for is in the closed
    dictionary. It will return True if it is and False if it isn't.
    """

    def state_in_closed(state_we_are_looking_for, closed_queue):
        # loop through the closed queue
        for closed_state in closed_queue:
            # check if the current closed state is the state we are looking for
            if (state_we_are_looking_for == closed_state):
                # return true
                return True

        # return false if not found
        return False

    """
    This function will grab a path from a queue based on the state
    """

    def grab_the_path(state, queue):
        # set the path to none for now
        queue_path = None

        # loop through the queue
        for queue_state in queue:
            # check if the path is not empty
            if (queue_path != None):
                # check if the queue state is the state we are looking for
                if (state == queue_state):
                    # get the state's path
                    queue_path = queue.get(state)

                    # return the path
                    return queue_path
            # if the path is empty
            else:
                # store a random path
                queue_path = queue.get(queue_state)

        # return the path
        return queue_path

    """
    This function reverses the path we want to reverse. It will pass directions
    in the path to reverse_the_direction() and stores it into a list. From that
    list, we flip the list backwards to make the path truly reversed.
    """

    def reverse_the_path(path_we_want_to_reverse):
        # create a list of reversed directions
        reverse_directions = []

        # loop through the path we want to reverse
        for direction in path_we_want_to_reverse:
            # at the current element, reverse the direction
            new_direction = reverse_the_direction(direction)

            # store the reversed direction into the list
            reverse_directions.extend([new_direction])

        # reverse the list (this makes the path backwards)
        reverse_directions.reverse()

        # store the reversed list
        reverse_the_path = reverse_directions

        # return the reversed path
        return reverse_the_path

    """
    This function reverses the direction we pass. It compares the passed
    direction with all 4 directions. If one of the directions matches,
    then we return the opposite direction of the passed direction. For
    example, if we pass the direction "North," then we would return
    the direction "South."
    """

    def reverse_the_direction(direction):
        # create the four directions Pacman can take
        direction_NORTH = "North"
        direction_EAST = "East"
        direction_WEST = "West"
        direction_SOUTH = "South"

        # check if our current direction is north
        if (direction == direction_NORTH):
            # return south
            return direction_SOUTH

        # check if our current direction is east
        elif (direction == direction_EAST):
            # return west
            return direction_WEST

        # check if our current direction is west
        elif (direction == direction_WEST):
            # return east
            return direction_EAST

        # check if our current direction is south
        elif (direction == direction_SOUTH):
            # return north
            return direction_NORTH

    """
    This function is the main "meet in the middle" algorithm. The pseudocode from the paper was implemented
    and altered to match our project's needs. It will call the helper functions above at the correct places
    and times as well.
    """

    def bidirectional_main():
        # store the start and goal states of the problem
        start_state = problem.getStartState()
        goal_state = problem.get_goal_state()

        # Forward Queue

        # create a priority queue where we push the start state, empty list, zero value, and heuristic into it
        open_forward = util.PriorityQueue()
        open_forward.push((start_state, [], 0), heuristic(start_state, problem))

        # create a dictionary where we will be storing g_values of the forward queue
        # (a complex data structure is not needed since we are only storing 1 value type)
        g_forward = dict()
        # set the g_value of the start state to 0
        g_forward[start_state] = 0

        # create a dictionary where we will be storing states and paths from open_forward priority queue
        # (a complex data structure is not needed since we are only storing a state and path)
        closed_forward = dict()

        # Backward Queue

        # create a priority queue where we push the goal state, empty list, zero value, and heuristic into it
        open_backward = util.PriorityQueue()
        open_backward.push((goal_state, [], 0), heuristic(goal_state, problem))

        # create a dictionary where we will be storing g_values of the forward queue
        # (a complex data structure is not needed since we are only storing 1 value type)
        g_backward = dict()
        # set the g_value of the goal state to 0
        g_backward[goal_state] = 0

        # create a dictionary where we will be storing states and paths from open_backward priority queue
        # (a complex data structure is not needed since we are only storing a state and path)
        closed_backward = dict()

        # set the epsilon value to 1
        epsilon_value = 1

        # set the U value to infinty for now
        infinity_value = float("inf")
        U = infinity_value

        # loop until both queues are empty
        while ((open_forward.isEmpty() == False) and (open_backward.isEmpty() == False)):

            # get the min f values of forward and backward searches
            minimum_f_value_forward = min_f_value(open_forward.heap, g_forward)
            minimum_f_value_backward = min_f_value(open_backward.heap, g_backward)

            # get the min g values of forward and backward searches
            minimum_g_value_forward = min_g_value(open_forward.heap, g_forward)
            minimum_g_value_backward = min_g_value(open_backward.heap, g_backward)

            # get the minimum priority of forward and backward searches
            minimum_priority_forward = min(minimum_f_value_forward, ((2) * (minimum_g_value_forward)))
            minimum_priority_backward = min(minimum_f_value_backward, ((2) * (minimum_g_value_backward)))

            # We calculate C by finding the minimum value between the minimum priorities of both queues.
            C = min(minimum_priority_forward, minimum_priority_backward)

            # We add the g_values of both queues and epsilon
            total_g_value = ((minimum_g_value_forward) + (minimum_g_value_backward) + (epsilon_value))

            # check if U is less than or equal to the max value of these 4 variables
            if (U <= max(C, minimum_f_value_forward, minimum_f_value_backward, total_g_value)):
                # check if the forward state is in closed backward dictionary
                path_closed_backward_flag = state_in_closed(state_forward, closed_backward)

                # check if the backward state is in the closed forward dictionary
                path_closed_forward_flag = state_in_closed(state_backward, closed_forward)

                # check if the forward and backward states are the same (signifies that both searches met in the middle)
                states_are_same = (state_forward == state_backward)

                # check if path closed backward flag is true
                if (path_closed_backward_flag == True):
                    # grab the backward path from the closed_backward dictionary
                    grab_backward_path = grab_the_path(state_forward, closed_backward)

                    # reverse the backward path
                    reverse_path_backward = reverse_the_path(grab_backward_path)

                    # add the reversed backward path to the forward path
                    complete_path = ((path_forward) + (reverse_path_backward))

                    # return the completed path
                    return complete_path

                # check if path closed forward flag is true
                elif (path_closed_forward_flag == True):
                    # grab the forward path from the closed_forward dictionary
                    grab_forward_path = grab_the_path(state_backward, closed_forward)

                    # reverse the backward path
                    reverse_path_backward = reverse_the_path(path_backward)

                    # add the reversed backward path to the forward path
                    complete_path = ((grab_forward_path) + (reverse_path_backward))

                    # return the completed path
                    return complete_path

                # check if states flag is true
                elif (states_are_same == True):
                    # reverse the backward path
                    reverse_path_backward = reverse_the_path(path_backward)

                    # add the reversed backward path to the forward path
                    complete_path = ((path_forward) + (reverse_path_backward))

                    # return the completed path
                    return complete_path

                # if the flags above are all false
                else:
                    # do nothing (set this variable to None cause it won't do anything to it)
                    complete_path = None

            # check if C is equal to the minimum priority of the forward queue
            if (C == minimum_priority_forward):
                # get the state, path, and g value of the node we are looking for
                state_forward, path_forward, g_value_forward = find_node(open_forward.heap, g_forward,
                                                                         minimum_priority_forward,
                                                                         minimum_g_value_forward)

                # store the state and path of the forward node into the closed_forward dictionary
                closed_forward[state_forward] = path_forward

                # grab the children of the forward node
                children_forward = problem.getSuccessors(state_forward)

                # go through every single child the forward node has
                for child in children_forward:
                    # store the state of the child
                    child_state = child[0]

                    # store the path to the child
                    child_path = [child[1]]

                    # store the g_value of the child
                    child_g_value = child[2]

                    # check if the state of the child is in the open_forward queue
                    open_forward_flag = state_in_list(open_forward.heap, child_state)

                    # check if the state of the child is in the closed_forward dictionary
                    closed_forward_flag = state_in_closed(child_state, closed_forward)

                    # add the g_value of the forward node and the g_value of the child
                    g_value_forward_child = ((g_value_forward) + (child_g_value))

                    # check if either flag is true
                    if ((open_forward_flag == True) or (closed_forward_flag == True)):
                        # grab the forward child g value
                        g_forward_child = grab_the_path(child_state, g_forward)

                        # check if the g_value of the child in the dictionary is less than the g_value of the child
                        if (g_forward_child <= g_value_forward_child):
                            # continue on
                            continue

                    # check if the open forward flag is true
                    if (open_forward_flag == True):
                        # remove the child from the open_forward queue
                        state_to_remove(open_forward.heap, child_state)

                    # check if the closed forward flag is true
                    if (closed_forward_flag == True):
                        # remove the child from the closed_forward dictionary
                        closed_forward.pop(child_state)

                    # store the calculated g_value of the child into g_forward
                    g_forward[child_state] = g_value_forward_child

                    # store the calculated g_value
                    g_f_child_value = g_value_forward_child

                    # calculate the f_value of the child by adding the heuristic and g_value of the child
                    f_value = ((heuristic(child_state, problem)) + (g_f_child_value))

                    # multiply the g_value of the child by 2
                    product_variable = ((2) * (g_f_child_value))

                    # get the priority of the child from f_value and the product
                    priority_forward = max(f_value, product_variable)

                    # calculate the new child path by adding the child path to the forward path
                    new_child_path = path_forward + child_path

                    # push the child into the open forward queue
                    open_forward.push((child_state, new_child_path, priority_forward), priority_forward)

                    # check if the state of the child is in the open backward queue
                    open_backward_flag = state_in_list(open_backward.heap, child_state)

                    # check if the open backward flag is true
                    if (open_backward_flag == True):
                        # grab the forward child g value
                        g_forward_child_value = grab_the_path(child_state, g_forward)

                        # get the minimum value of U and g_value of the child and store it into U
                        U = min(U, g_forward_child_value)



            # check if C is equal to the minimum priority of the backward queue
            elif (C == minimum_priority_backward):
                # get the state, path, and g value of the node we are looking for
                state_backward, path_backward, g_value_backward = find_node(open_backward.heap, g_backward,
                                                                            minimum_priority_backward,
                                                                            minimum_g_value_backward)

                # store the state and path of the backward node into the closed_backward dictionary
                closed_backward[state_backward] = path_backward

                # grab the children of the backward node
                children_backward = problem.getSuccessors(state_backward)

                # go through every single child the backward node has
                for child in children_backward:
                    # store the state of the child
                    child_state = child[0]

                    # store the path to the child
                    child_path = [child[1]]

                    # store the g_value of the child
                    child_g_value = child[2]

                    # check if the state of the child is in the open_backward queue
                    open_backward_flag = state_in_list(open_backward.heap, child_state)

                    # check if the state of the child is in the closed_backward dictionary
                    closed_backward_flag = state_in_closed(child_state, closed_backward)

                    # add the g_value of the backward node and the g_value of the child
                    g_value_backward_child = ((g_value_backward) + (child_g_value))

                    # check if either flag is true
                    if ((open_backward_flag == True) or (closed_backward_flag == True)):
                        # grab the backward child g value
                        g_backward_child = grab_the_path(child_state, g_backward)

                        # check if the g_value of the child in the dictionary is less than the g_value of the child
                        if (g_backward_child <= g_value_backward_child):
                            # continue on
                            continue

                    # check if the open backward flag is true
                    if (open_backward_flag == True):
                        # remove the child from the open_backward queue
                        state_to_remove(open_backward.heap, child_state)

                    # check if the closed backward flag is true
                    if (closed_backward_flag == True):
                        # remove the child from the closed_backward dictionary
                        closed_backward.pop(child_state)

                    # store the calculated g_value of the child into g_backward
                    g_backward[child_state] = g_value_backward_child

                    # store the calculated g_value of the child
                    g_b_child_value = g_value_backward_child

                    # calculate the f_value of the child by adding the heuristic and g_value of the child
                    f_value = ((heuristic(child_state, problem)) + (g_b_child_value))

                    # multiply the g_value of the child by 2
                    product_variable = ((2) * (g_b_child_value))

                    # get the priority of the child from f_value and the product
                    priority_backward = max(f_value, product_variable)

                    # calculate the new child path by adding the child path to the backward path
                    new_child_path = path_backward + child_path

                    # push the child into the open backward queue
                    open_backward.push((child_state, new_child_path, priority_backward), priority_backward)

                    # check if the state of the child is in the open forward queue
                    open_forward_flag = state_in_list(open_forward.heap, child_state)

                    # check if the open forward flag is true
                    if (open_forward_flag == True):
                        # grab the backward child g value
                        g_backward_child_value = grab_the_path(child_state, g_backward)

                        # get the minimum value of U and g_value of the child and store it into U
                        U = min(U, g_backward_child_value)

        # return an empty list
        return []

    # get the path from the main function
    get_meet_in_the_middle_path = bidirectional_main()

    # return the path
    return get_meet_in_the_middle_path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
mm0 = MM0Search
mm = bidirectional_MM