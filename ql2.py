#attempt to adapt ql1 for sub-goal by applying q learning twice, did not work

import numpy as np
import random
from maze1 import Maze

#q learning params
maze_size = 10
num_actions = 4 
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.2 
episodes = 5000

#action mappings
actions = {0: (-1, 0),
           1: (1, 0),  
           2: (0, -1), 
           3: (0, 1)} 

#init q table
Q_table = np.zeros((maze_size, maze_size, num_actions))

#get next state given s,a
def next_state(state, action):
    x, y = state
    dx, dy = actions[action]
    new_state = (x + dx, y + dy)

    #check bounds
    if 0 <= new_state[0] < maze_size and 0 <= new_state[1] < maze_size:
        #check for wall
        if maze[new_state[0], new_state[1]] == 0:
            return new_state
    return state


def q_learning(maze, start, goal):
    global Q_table
    for episode in range(episodes):
        state = start
        done = False
        while not done:
            #choose action using e-greedy
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, num_actions - 1)  #explore
            else:
                action = np.argmax(Q_table[state[0], state[1]])  #exploit

            #get next state & reward
            next_state_pos = next_state(state, action)
            if next_state_pos == goal:
                reward = 10  #goal
                done = True
            elif maze[next_state_pos[0], next_state_pos[1]] == 1:  #wall
                reward = -1
            else:
                reward = -0.1 

            #update
            old_value = Q_table[state[0], state[1], action]
            next_max = np.max(Q_table[next_state_pos[0], next_state_pos[1]])
            new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
            Q_table[state[0], state[1], action] = new_value

            
            state = next_state_pos #get next state

#maze stuff
maze = Maze()
maze_with_symbols = maze.__getfinalmaze__()
start = maze.__getstart__()
sub_goal = maze.__getsubgoal__()
end_goal = maze.__getendgoal__()
maze = maze.__getmaze__()

print("Generated Maze:")
for row in maze_with_symbols:
    print(" ".join(map(str, row)))


#q learning from S to G
q_learning(maze, start, sub_goal)

#QL from G to E
q_learning(maze, sub_goal, end_goal)

#get optimal path
def find_optimal_path(maze, start, goal):
    path = []
    state = start
    while state != goal:
        path.append(state)
        action = np.argmax(Q_table[state[0], state[1]])
        state = next_state(state, action)
    path.append(goal)
    return path

optimal_path_s_to_g = find_optimal_path(maze, start, sub_goal)
optimal_path_g_to_e = find_optimal_path(maze, sub_goal, end_goal)
print("\nOptimal path from start to sub-goal:", optimal_path_s_to_g)
print("Optimal path from sub-goal to end-goal:", optimal_path_g_to_e)
