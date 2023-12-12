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

from asyncio.windows_events import NULL
from collections import deque
from dis import Instruction
from pickle import TRUE
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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
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

    currentstate = [[problem.getStartState(), NULL], None, 0] #The list looks like [[coordinates, direction, cost], parent]

    visited = []
    stack = deque()

    visited.append(currentstate[0][0])
    stack.append(currentstate)

    while stack: #due to visited the goal can only be reached once
        currentstate = stack.pop()

        if problem.isGoalState(currentstate[0][0]):
            break

        for neighbour in problem.getSuccessors(currentstate[0][0]):
            if neighbour[0] not in visited or problem.isGoalState(neighbour[0]):
                visited.append(neighbour[0])
                stack.append([neighbour, currentstate]) #Moved to the top of the stack so each branch is searched in depth, so last in first serve

    instructions = []

    while currentstate[1] != None: #Until parent is None, so the startpos
        instructions.insert(0,currentstate[0][1]) #The instructions are read from the goal, otherwise we use reverse
        currentstate = currentstate[1]

    return instructions

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    currentstate = [[problem.getStartState(), NULL, 0], None] #The list looks like [[coordinates, direction, cost], parent]

    visited = []
    queue = deque()

    visited.append(currentstate[0][0])
    queue.append(currentstate)

    while queue: #due to visited the goal can only be reached once
        currentstate = queue.pop()

        if problem.isGoalState(currentstate[0][0]):
            break

        for neighbour in problem.getSuccessors(currentstate[0][0]):
            if neighbour[0] not in visited or problem.isGoalState(neighbour[0]):
                visited.append(neighbour[0])
                queue.appendleft([neighbour, currentstate]) #Appending left lets the queue prioritize the higher layers first, so first in first serve

    instructions = []

    while currentstate[1] != None: #Until parent is None, so the startpos
        instructions.insert(0,currentstate[0][1]) #The instructions are read from the goal, otherwise we use reverse
        currentstate = currentstate[1]

    return instructions

    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    def getcost (item): #returns cost of node
        return item[0][2]

    currentstate = [[problem.getStartState(), NULL, 0], None] #The list looks like [[coordinates, direction, cost], parent]

    visited = []
    queue = deque()

    routes = []

    visited.append(currentstate[0][0])
    queue.append(currentstate)

    while queue: #due to visited the goal can only be reached once
        queue = sorted(queue, key=getcost, reverse=True) #sort queue on lowest cost
        currentstate = queue.pop()

        if problem.isGoalState(currentstate[0][0]):
            break

        for neighbour in problem.getSuccessors(currentstate[0][0]):
            if neighbour[0] not in visited or problem.isGoalState(neighbour[0]):
                visited.append(neighbour[0])
                neighbourlist = list(neighbour) #cast to list, because tuples can't have item assignment
                neighbourlist[2] += currentstate[0][2]
                queue.append([neighbourlist, currentstate])

    instructions = []

    while currentstate[1] != None: #Until parent is None, so the startpos
        instructions.insert(0,currentstate[0][1]) #The instructions are read from the goal, otherwise we use reverse
        currentstate = currentstate[1]

    return instructions

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    def getcost (item): #returns cost of node
        totalcost = item[0][2] + heuristic(item[0][0], problem) # f = g* + h
        return totalcost

    currentstate = [[problem.getStartState(), NULL, 0], None] #The list looks like [[coordinates, direction, cost], parent]

    visited = []
    queue = deque()

    routes = []

    visited.append(currentstate[0][0])
    queue.append(currentstate)

    while queue: #due to visited the goal can only be reached once
        queue = sorted(queue, key=getcost, reverse=True) #sort queue on lowest cost
        currentstate = queue.pop()

        if problem.isGoalState(currentstate[0][0]):
            break

        for neighbour in problem.getSuccessors(currentstate[0][0]):
            if neighbour[0] not in visited or problem.isGoalState(neighbour[0]):
                visited.append(neighbour[0])
                neighbourlist = list(neighbour) #cast to list, because tuples can't have item assignment
                neighbourlist[2] += currentstate[0][2]
                queue.append([neighbourlist, currentstate])

    instructions = []

    while currentstate[1] != None: #Until parent is None, so the startpos
        instructions.insert(0,currentstate[0][1]) #The instructions are read from the goal, otherwise we use reverse
        currentstate = currentstate[1]

    return instructions

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
