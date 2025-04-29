import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from queue import PriorityQueue

# Robot and workspace dimensions (simplified)
ARM_LENGTH = 3.0
ARM_WIDTH = 0.2
WORKSPACE_WIDTH = 8.0
WORKSPACE_HEIGHT = 4.0

# Simplified obstacles (pipes)
OBSTACLES = [
    {'x': 2.0, 'y': 2.0, 'radius': 0.3},
    {'x': 4.0, 'y': 2.0, 'radius': 0.3},
    {'x': 6.0, 'y': 2.0, 'radius': 0.3}
]

# Simplified bins
RED_BIN = {'x': 3.0, 'y': 3.0, 'width': 0.8, 'height': 0.2}
BLUE_BIN = {'x': 5.0, 'y': 3.0, 'width': 0.8, 'height': 0.2}

# Configuration space discretization (simplified)
X_RESOLUTION = 50
ALPHA_RESOLUTION = 50

# Configuration space ranges
X_MIN, X_MAX = 0, WORKSPACE_WIDTH - ARM_LENGTH
ALPHA_MIN, ALPHA_MAX = 0, np.pi/2

# Create the configuration space grid
x_values = np.linspace(X_MIN, X_MAX, X_RESOLUTION)
alpha_values = np.linspace(ALPHA_MIN, ALPHA_MAX, ALPHA_RESOLUTION)
config_space = np.zeros((ALPHA_RESOLUTION, X_RESOLUTION))

def check_collision(x, alpha):
    """Simple collision check for the arm at a given configuration"""
    # Sample points along the arm
    num_points = 10
    arm_x = np.linspace(x, x + ARM_LENGTH * np.cos(alpha), num_points)
    arm_y = np.linspace(0, ARM_LENGTH * np.sin(alpha), num_points)
    
    # Check each point for collision with obstacles
    for i in range(num_points):
        for obstacle in OBSTACLES:
            dist = np.sqrt((arm_x[i] - obstacle['x'])**2 + (arm_y[i] - obstacle['y'])**2)
            if dist <= obstacle['radius'] + ARM_WIDTH/2:
                return True  # Collision detected
    
    return False  # No collision

# Build the configuration space
print("Building configuration space...")
for i, alpha in enumerate(alpha_values):
    for j, x in enumerate(x_values):
        if check_collision(x, alpha):
            config_space[i, j] = 1  # Mark as obstacle
print("Configuration space built!")

# Find configurations for reaching the bins
def get_bin_configuration(bin_info):
    """Find a valid configuration to reach the bin"""
    bin_center_x = bin_info['x'] + bin_info['width'] / 2
    bin_center_y = bin_info['y'] + bin_info['height'] / 2
    
    # Try configurations that might reach the bin
    best_config = None
    best_distance = float('inf')
    
    for i, alpha in enumerate(alpha_values):
        for j, x in enumerate(x_values):
            if config_space[i, j] == 0:  # If collision-free
                # End effector position
                end_x = x + ARM_LENGTH * np.cos(alpha)
                end_y = ARM_LENGTH * np.sin(alpha)
                
                # Distance to bin center
                dist = np.sqrt((end_x - bin_center_x)**2 + (end_y - bin_center_y)**2)
                
                if dist < best_distance:
                    best_distance = dist
                    best_config = (i, j)
    
    return best_config

# Find start and end configurations
start_config = get_bin_configuration(RED_BIN)
goal_config = get_bin_configuration(BLUE_BIN)

print(f"Start config: alpha={alpha_values[start_config[0]]:.2f}, x={x_values[start_config[1]]:.2f}")
print(f"Goal config: alpha={alpha_values[goal_config[0]]:.2f}, x={x_values[goal_config[1]]:.2f}")

# Simplified shortest path using Dijkstra's algorithm
def find_shortest_path(start, goal, config_space):
    """Find the shortest path between start and goal configurations"""
    # Initialize distances
    rows, cols = config_space.shape
    distances = np.full((rows, cols), np.inf)
    distances[start] = 0
    
    # Initialize priority queue and visited set
    pq = PriorityQueue()
    pq.put((0, start))
    visited = set()
    
    # Track path
    came_from = {}
    
    while not pq.empty():
        current_dist, current = pq.get()
        
        if current in visited:
            continue
        
        if current == goal:
            break
        
        visited.add(current)
        
        # Check 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds and obstacles
                if (0 <= neighbor[0] < rows and 
                    0 <= neighbor[1] < cols and 
                    config_space[neighbor] == 0 and
                    neighbor not in visited):
                    
                    # Calculate distance (Euclidean)
                    move_cost = np.sqrt(dx**2 + dy**2)
                    new_dist = distances[current] + move_cost
                    
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        came_from[neighbor] = current
                        pq.put((new_dist, neighbor))
    
    # Reconstruct path
    if goal in came_from or goal == start:
        path = [goal]
        current = goal
        
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        return path[::-1]  # Reverse to get start-to-goal
    else:
        return None  # No path found

print("Finding shortest path...")
path = find_shortest_path(start_config, goal_config, config_space)

if path:
    print(f"Path found with {len(path)} steps!")
    
    # Convert path indices to actual configurations
    config_path = [(alpha_values[i], x_values[j]) for i, j in path]
    
    # Visualization
    plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    # Plot configuration space
    ax1 = plt.subplot(gs[0])
    ax1.imshow(config_space, cmap='gray_r', origin='lower', 
               extent=[X_MIN, X_MAX, ALPHA_MIN, ALPHA_MAX], aspect='auto')
    
    # Plot path in configuration space
    path_x = [x_values[p[1]] for p in path]
    path_alpha = [alpha_values[p[0]] for p in path]
    ax1.plot(path_x, path_alpha, 'r-', linewidth=2)
    ax1.plot(path_x[0], path_alpha[0], 'go', markersize=8)  # Start
    ax1.plot(path_x[-1], path_alpha[-1], 'bo', markersize=8)  # Goal
    
    ax1.set_xlabel('x position')
    ax1.set_ylabel('α (rad)')
    ax1.set_title('Configuration Space with Shortest Path')
    
    # Prepare workspace visualization
    ax2 = plt.subplot(gs[1])
    
    # Function to visualize the robot at a configuration
    def visualize_robot(ax, x, alpha):
        # Draw workspace
        ax.set_xlim(0, WORKSPACE_WIDTH)
        ax.set_ylim(0, WORKSPACE_HEIGHT)
        ax.plot([0, WORKSPACE_WIDTH], [0, 0], 'k-', linewidth=2)
        
        # Draw bins
        ax.add_patch(Rectangle((RED_BIN['x'], RED_BIN['y']), 
                              RED_BIN['width'], RED_BIN['height'], 
                              color='red', alpha=0.7))
        ax.add_patch(Rectangle((BLUE_BIN['x'], BLUE_BIN['y']), 
                              BLUE_BIN['width'], BLUE_BIN['height'], 
                              color='blue', alpha=0.7))
        
        # Draw obstacles
        for obstacle in OBSTACLES:
            ax.add_patch(Circle((obstacle['x'], obstacle['y']), 
                               obstacle['radius'], color='gray'))
        
        # Draw robot arm
        end_x = x + ARM_LENGTH * np.cos(alpha)
        end_y = ARM_LENGTH * np.sin(alpha)
        
        # Simple arm drawing
        ax.plot([x, end_x], [0, end_y], 'c-', linewidth=ARM_WIDTH*20)
        ax.plot(x, 0, 'ks', markersize=10)  # Base
        ax.plot(end_x, end_y, 'bo', markersize=8)  # End effector
    
    # Animation function
    def update(frame):
        ax2.clear()
        
        if frame < len(config_path):
            alpha, x = config_path[frame]
        else:
            alpha, x = config_path[-1]
            
        visualize_robot(ax2, x, alpha)
        ax2.set_title(f'Robot Workspace: Step {frame}/{len(config_path)-1}')
        
        return []
    
    # Create animation
    ani = FuncAnimation(plt.gcf(), update, frames=len(config_path), 
                        interval=100, blit=True)
    
    # Display key frames
    plt.tight_layout()
    
    # Show five snapshots of the path
    frame_indices = [0, len(path)//4, len(path)//2, 3*len(path)//4, len(path)-1]
    
    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(frame_indices):
        ax = plt.subplot(1, 5, i+1)
        alpha, x = config_path[idx]
        visualize_robot(ax, x, alpha)
        ax.set_title(f'Step {idx}')
        
        # Only show labels on first subplot
        if i > 0:
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization complete!")
else:
    print("No path found!")