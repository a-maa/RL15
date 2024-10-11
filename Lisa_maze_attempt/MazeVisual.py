# Base imports
import numpy as np                  # Library for arrays, matrices, mathematical functions, etc.
import matplotlib.pyplot as plt     # Good for visualisations.


# Initialising the maze
class Maze:
    def __init__(self, maze, start_position, goal_position, sub_goal_position):
        self.maze = maze

        # Later necessary for training
        self.maze_width = maze_layout.shape[1]          # rows of maze, also knows as x-azis.
        self.maze_height = maze_layout.shape[0]         # columns of maze, also known as y-axis.

        self.start_position = start_position            # start position - S.
        self.goal_position = goal_position              # end goal position - E.
        self.sub_goal_position = sub_goal_position      # sub goal position - G.

    def show_maze(self):
        plt.figure(figsize=(5,5))

        plt.imshow(self.maze, cmap='Pastel1_r')

        # Placements for the start, end, and sub goal positions.
        plt.text(self.start_position[0], self.start_position[1], 'S', ha='center', va='center', color='green', fontsize=15)
        plt.text(self.goal_position[0], self.goal_position[1], 'E', ha='center', va='center', color='red', fontsize=15)
        plt.text(self.sub_goal_position[0], self.sub_goal_position[1], 'G', ha='center', va='center', color='blue', fontsize=15)

        # Hide the digits and labels from the plot visualistion. 
        plt.xticks([]), plt.yticks([])

        # Ensures the plot will visualise when running the code.
        plt.show()


# The layout of the 10x10 maze:
# 1 = wall.
# 0 = open area.
# (If wanted, we can later make another file where we can generate larger mazes like 100x100)
maze_layout = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# Places the start, end, and sub goal at the correct coordinates (rows, columns).
maze = Maze(maze_layout, (1, 1), (7, 8), (3,5))

# Actually visualises the plot with matplotlib.
maze.show_maze()