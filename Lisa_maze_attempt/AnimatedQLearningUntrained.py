import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from MazeVisual import maze
from Agent import QLearningAgent, actions

# Reward system
end_reward = 100
sub_reward = 50
wall_penalty = -10
step_penalty = -1

# Agent's facing directions.
direction_symbols = {
    (0, 1): ">",   # Right.
    (0, -1): "<",  # Left.
    (-1, 0): "^",  # Up.
    (1, 0): "v"    # Down.
}

def finish_episode(agent, maze, current_episode, train=True):
    current_state = maze.start_position
    end_reached = False
    final_reward = 0
    episode_step = 0
    path = [current_state]
    sub_goal_reached = False

    while not end_reached:
        action = agent.get_action(current_state, current_episode)
        next_state = (current_state[0] + actions[action][0], current_state[1] + actions[action][1])

        if next_state[0] < 0 or next_state[0] >= maze.maze_height or next_state[1] < 0 or next_state[1] >= maze.maze_width or maze.maze[next_state[1]][next_state[0]] == 1:
            reward = wall_penalty
            next_state = current_state
        elif next_state == maze.sub_goal_position:
            reward = sub_reward
            sub_goal_reached = True
        elif next_state == maze.goal_position:
            if sub_goal_reached:
                reward = end_reward
                end_reached = True
            else:
                reward = wall_penalty
                next_state = current_state
        else:
            reward = step_penalty

        final_reward += reward
        episode_step += 1

        if train:
            agent.update_q_table(current_state, action, next_state, reward)

        current_state = next_state
        path.append(current_state)

    return final_reward, episode_step, path, sub_goal_reached


def animate_agent(agent, maze, num_episodes=1):
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plot the initial maze positions.
    ax.imshow(maze.maze, cmap='Pastel1')
    ax.text(maze.start_position[0], maze.start_position[1], 'S', ha='center', va='center', color='green', fontsize=15)
    ax.text(maze.goal_position[0], maze.goal_position[1], 'E', ha='center', va='center', color='red', fontsize=15)
    sub_goal_marker = ax.text(maze.sub_goal_position[0], maze.sub_goal_position[1], 'G', ha='center', va='center', color='blue', fontsize=15)
    
    # Initialize the agent's position and direction marker.
    agent_marker = ax.text(maze.start_position[0], maze.start_position[1], ">", ha='center', va='center', fontsize=15, color='white')  
    path_markers = []

    episode_reward, episode_step, path, sub_goal_reached = finish_episode(agent, maze, num_episodes, train=False)

    def update(frame):
        if frame < len(path) - 1:
            pos = path[frame]
            next_pos = path[frame + 1]
            movement = (next_pos[0] - pos[0], next_pos[1] - pos[1])  # Calculate direction.

            # Update agent marker with direction symbol based on movement
            agent_marker.set_position((pos[0], pos[1]))

            # Default to 'o' if no direction considered.
            agent_marker.set_text(direction_symbols.get(movement, "o"))  

            # Change the "G" sub-goal marker from blue to black when reached.
            if pos == maze.sub_goal_position:
                sub_goal_marker.set_color('black')

        # If it's the last frame (goal reached), plot the entire path.
        if frame == len(path) - 1:
            for (x, y) in path:
                marker, = ax.plot(x, y, "o", color='white', markersize=5)
                path_markers.append(marker)

        return agent_marker, sub_goal_marker, *path_markers

    # Animates the agent going through the maze. Change "interval" to increase its speed; less for faster, more for slower.
    anim = FuncAnimation(fig, update, frames=len(path), interval=50, blit=True, repeat=False)

    # Shows the step sum and total reward after closing the animation. Same as in the unanimated variant. 
    print("Final Path:")
    for row, col in path:
        print(f"({row}, {col})-> ", end='')
    print("End Reached")

    print("Total steps:", episode_step)
    print("Total reward:", episode_reward)

    plt.xticks([]), plt.yticks([])
    plt.grid(color='black', linewidth=2)
    plt.show()

    return episode_step, episode_reward

agent = QLearningAgent(maze)
animate_agent(agent, maze)
