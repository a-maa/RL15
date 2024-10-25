#attempt to adapt ql1 by adding extra logic in q learning method - missing logic, don't run

import numpy as np
import random
from maze1 import Maze

#q learning params
maze_size = 10
num_actions = 4  #UDLR
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.2 
episodes = 5000

#actions
actions = {0: (-1, 0),  #u
           1: (1, 0),   #d
           2: (0, -1),  #l
           3: (0, 1)}   #r

#init q table
Q_table = np.zeros((maze_size, maze_size, num_actions))

#get next pos given (s,a)
def next_state(state, action):
    x, y = state
    dx, dy = actions[action]
    new_state = (x + dx, y + dy)

    #check bounds
    if 0 <= new_state[0] < maze_size and 0 <= new_state[1] < maze_size:
        #check if state is wall
        if maze[new_state[0], new_state[1]] == 0:
            return new_state
    return state  #if invalid move


#algo implementation
def q_learning(maze, start, sub_goal, end_goal):
    global Q_table
    for episode in range(episodes):
        state = start
        done = False
        g_reached = False
        while not done:
            #choose action using e-greedy
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, num_actions - 1)  #explore
            else:
                action = np.argmax(Q_table[state[0], state[1]])  #exploit

            #get next state & reward
            next_state_pos = next_state(state, action)
            if (next_state_pos == end_goal) and g_reached:
                reward = 10  #goal
                done = True
            elif (next_state_pos == sub_goal) and not g_reached:
                g_reached = True
                reward = 5
            elif maze[next_state_pos] == 1:  #wall
                reward = -1
            else:
                reward = -0.1  #decrease reward for each move

            #update
            old_value = Q_table[state[0], state[1], action]
            next_max = np.max(Q_table[next_state_pos[0], next_state_pos[1]])
            new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
            Q_table[state[0], state[1], action] = new_value

            #move to next state
            state = next_state_pos


maze = Maze()
maze_with_symbols = maze.__getfinalmaze__()
start = maze.__getstart__()
sub_goal = maze.__getsubgoal__()
end_goal = maze.__getendgoal__()
maze = maze.__getmaze__()

print("Generated Maze:")
for row in maze_with_symbols:
    print(" ".join(map(str, row)))

#run q learning from start to end
q_learning(maze, start, sub_goal, end_goal)


def find_optimal_path(maze, start, end_goal):
    path = []
    state = start
    while state != end_goal:
        path.append(state)
        action = np.argmax(Q_table[state[0], state[1]])
        state = next_state(state, action)
    path.append(end_goal)
    return path


optimal_path = find_optimal_path(maze, start, sub_goal)
print("\nOptimal path from start to end goal:", optimal_path)
