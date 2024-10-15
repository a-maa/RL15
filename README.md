# **Q-Learning Maze**

*This project is based on an assignment given by Vrije Universiteit Amsterdam.*

### **Project Overview:**

In this project, you will design and implement reinforcement learning (RL) algorithms
to solve a 10 × 10 maze with a sub-goal. The maze consists of walls and open
spaces, where the agent (player) starts at a specific point and needs to navigate
through the maze to reach the final goal. A sub-goal is introduced in the maze to
promote intermediate progress, requiring you to adjust your RL algorithms to first
reach this sub-goal before advancing to the final goal.

### **Group Notes**

For this project, the choice was given between multiple different RL algorithms. 
In this case Q-Learning was chosen mainly due to its effectiveness when working 
in less complex environments. in a 10 x 10 maze, this would mean it wouldn't pose
too much of an issue.

the maze has been build with matplotlib through a set of arrays which would state
if something is a wall (1) or an open space (0). As the overview mentioned before,
there is a starting points (S), a sub-goal (G), and an ending point (E). It was
important to ensure that the agent trained with the Q-Learning algorithm would 
pass through the sub goal before reaching the ending point. Therefore, it is 
important that all the parameters as well as the rewards/penalties were chosen 
carefully to train the agent to the best ability. 

### **Current Materials Used**

As of now, the project has been built fully within Python. For ease, the code
has been written into jupyter notebook. 

The imports that have been used are NumPy for the mathematical functionality, 
MatPlotLib for the visualisation as well as to animate the path of the agent, 
and IPython.display to ensure the animation would work in jupyter notebook.

### **Where To Find The Important Files (currently)**

Inside folder LSecondAttempt, you can find two .ipynb files named 10x10 and 20x20.
The 10x10 notebook is the mandatory part of the project that will be handed in and 
analysed in the end. 20x20 is a side project for comparison which is a more advanced 
version to show the code of the sub-goal in a more comprehensive manner. 
