import numpy as np
import random

class Maze():
    """Attributes:
        maze - original maze with 0s and 1s representing paths and walls respectively
        start - start pos
        sub_goal - sub goal pos
        end_goal - end goal pos
        maze_with_symbols - final maze with S,G,E in place of the special pts"""
    
    def __init__(self) -> None:
        self.maze = np.ones((10, 10), dtype=int) #generate 10x10 grid of 1s

        self.start = (random.randint(0, 9), random.randint(0, 9))
        self.sub_goal = (random.randint(0, 9), random.randint(0, 9))
        self.end_goal = (random.randint(0, 9), random.randint(0, 9)) #random selection for special pts


        while self.sub_goal == self.start:
            self.sub_goal = (random.randint(0, 9), random.randint(0, 9))
        while self.end_goal == self.start or self.end_goal == self.sub_goal:
            self.end_goal = (random.randint(0, 9), random.randint(0, 9)) #re-define pts if they're the same

        
        self.maze[self.start] = 0
        self.maze[self.sub_goal] = 0
        self.maze[self.end_goal] = 0 #ensure pts are open to cross through

        #use dfs for solvable path
        def dfs(x, y):
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] #directions: r,l,u,d
            random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if 1 <= nx < 9 and 1 <= ny < 9 and self.maze[nx, ny] == 1:
                    #avoid loops
                    if self.maze[nx + dx, ny + dy] == 1:
                        self.maze[nx, ny] = 0
                        self.maze[nx + dx, ny + dy] = 0 #make open path
                        dfs(nx + dx, ny + dy) 

        #dfs from start pos
        dfs(self.start[0], self.start[1])

        #make open paths to goal & subgoal
        def connect_points(p1, p2):
            x1, y1 = p1
            x2, y2 = p2

            if x1 == x2:  #same row
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    self.maze[x1, y] = 0
            elif y1 == y2:  #same col
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    self.maze[x, y1] = 0
            else:  #zigzag path if diagonals
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    self.maze[x1, y] = 0
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    self.maze[x, y2] = 0

        #connect S to G and G to E
        connect_points(self.start, self.sub_goal)
        connect_points(self.sub_goal, self.end_goal)

        #generate mazwe
        self.maze_with_symbols = np.array(self.maze, dtype=object)
        self.maze_with_symbols[self.start] = 'S'
        self.maze_with_symbols[self.sub_goal] = 'G'
        self.maze_with_symbols[self.end_goal] = 'E'
    

    def __getstart__(self):
        return self.start
    
    def __getsubgoal__(self):
        return self.sub_goal
    
    def __getendgoal__(self):
        return self.end_goal
    
    def __getmaze__(self):
        return self.maze
    
    def __getfinalmaze__(self):
        return self.maze_with_symbols
    
