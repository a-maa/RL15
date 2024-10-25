import numpy as np
import random

def create_maze():
    maze = np.ones((10, 10), dtype=int) #generate 10x10 grid of 1s

    start = (random.randint(0, 9), random.randint(0, 9))
    sub_goal = (random.randint(0, 9), random.randint(0, 9))
    end_goal = (random.randint(0, 9), random.randint(0, 9)) #random selection for special pts


    while sub_goal == start:
        sub_goal = (random.randint(0, 9), random.randint(0, 9))
    while end_goal == start or end_goal == sub_goal:
        end_goal = (random.randint(0, 9), random.randint(0, 9)) #re-define pts if they're the same

    
    maze[start] = 0
    maze[sub_goal] = 0
    maze[end_goal] = 0 #ensure pts are open to cross through

    #use dfs for solvable path
    def dfs(x, y):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] #directions: r,l,u,d
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 1 <= nx < 9 and 1 <= ny < 9 and maze[nx, ny] == 1:
                #avoid loops
                if maze[nx + dx, ny + dy] == 1:
                    maze[nx, ny] = 0
                    maze[nx + dx, ny + dy] = 0 #make open path
                    dfs(nx + dx, ny + dy) 

    #dfs from start pos
    dfs(start[0], start[1])

    #make open paths to goal & subgoal
    def connect_points(p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        if x1 == x2:  #same row
            for y in range(min(y1, y2), max(y1, y2) + 1):
                maze[x1, y] = 0
        elif y1 == y2:  #same col
            for x in range(min(x1, x2), max(x1, x2) + 1):
                maze[x, y1] = 0
        else:  #zigzag path if diagonals
            for y in range(min(y1, y2), max(y1, y2) + 1):
                maze[x1, y] = 0
            for x in range(min(x1, x2), max(x1, x2) + 1):
                maze[x, y2] = 0

    #connect S to G and G to E
    connect_points(start, sub_goal)
    connect_points(sub_goal, end_goal)

    #generate mazwe
    maze_with_symbols = np.array(maze, dtype=object)
    maze_with_symbols[start] = 'S'
    maze_with_symbols[sub_goal] = 'G'
    maze_with_symbols[end_goal] = 'E'

    return maze_with_symbols


maze = create_maze()
for row in maze:
    print(" ".join(map(str, row)))
