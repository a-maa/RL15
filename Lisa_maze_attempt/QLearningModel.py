# Base imports
import numpy as np                  # Library for arrays, matrices, mathematical functions, etc.
import matplotlib.pyplot as plt     # Good for visualisations.
from MazeVisual import Maze, maze

# Reward system
end_reward = 100                    # Reward for reaching the end goal state.
sub_reward = 50                     # Reward for reaching the sub goal state.
wall_penalty = -10                  # Penalty for touching any wall in the maze.
step_penalty = -1                   # Penalty for taking any step in the maze.

# Actions the agent can take.
actions = [
   (-1, 0),                         # Moving one step up.
   (1, 0),                          # Moving one step down.
   (0, -1),                         # Moving one step left.
   (0, 1)                           # Moving one step right.
]

# Initialise the Q-Learning agent 
class QLearningAgent:
    def __init__(self, maze, learning_rate=0.1, discount_factor=0.9, exploration_start=1.0, exploration_end=0.01, num_episodes=100):
        self.q_table = np.zeros((maze.maze_height, maze.maze_width, 4)) 
        # 4 means up, down, left, right. 

        self.learning_rate = learning_rate          
        self.discount_factor = discount_factor      
        self.exploration_start = exploration_start  
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes

    # Calculates the rate of exploration to exploitation over time -> start with a lot of exploration and eventually prefer exploitation.
    def get_exploration_rate(self, current_episode):
        exploration_rate = self.exploration_start * (self.exploration_end / self.exploration_start) ** (current_episode / self.num_episodes)
        return exploration_rate
    
    # Chooses what movement action to make. 
    def get_action(self, state, current_episode):
        exploration_rate = self.get_exploration_rate(current_episode)

        # Select an action for the given state either randomly (exploration) or using the Q-table (exploitation).
        if np.random.rand() < exploration_rate:
            return np.random.randint(4) 
        else:
            #Chooses the action with the highest Q-value for the given state.
            return np.argmax(self.q_table[state]) 
        
    # Updates the Q-values in the Q-table based on its actions and states.
    def update_q_table(self, state, action, next_state, reward):
        best_next_action = np.argmax(self.q_table[next_state])

        current_q_value = self.q_table[state][action]

        # Formula to update the Q-value based on the theory of the Q-Learning algorithm.
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * self.q_table[next_state][best_next_action] - current_q_value)

        # Apply new Q-value for current action and state. 
        self.q_table[state][action] = new_q_value


