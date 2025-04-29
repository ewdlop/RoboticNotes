import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import cdist

# Robot and workspace dimensions
ARM_LENGTH = 4.0  # Length of the robot arm
ARM_WIDTH = 0.3   # Width of the robot arm
WORKSPACE_WIDTH = 10.0
WORKSPACE_HEIGHT = 5.0

# Obstacle definitions (gray pipes)
OBSTACLES = [
    {'x': 2.5, 'y': 2.5, 'radius': 0.4},
    {'x': 5.0, 'y': 2.5, 'radius': 0.4},
    {'x': 7.5, 'y': 2.5, 'radius': 0.4}
]

# Bins definition
RED_BIN = {'x': 3.5, 'y': 3.5, 'width': 1.0, 'height': 0.3}
BLUE_BIN = {'x': 6.0, 'y': 3.5, 'width': 1.0, 'height': 0.3}

# Configuration space parameters
ALPHA_MIN = 0  # Minimum arm angle (in radians)
ALPHA_MAX = np.pi/2  # Maximum arm angle (in radians)
X_MIN = 0
X_MAX = WORKSPACE_WIDTH - ARM_LENGTH  # Maximum x-position

# Discretization for configuration space
X_STEPS = 100
ALPHA_STEPS = 100

# Create configuration space grid
x_values = np.linspace(X_MIN, X_MAX, X_STEPS)
alpha_values = np.linspace(ALPHA_MIN, ALPHA_MAX, ALPHA_STEPS)
config_space = np.zeros((ALPHA_STEPS, X_STEPS))

def get_arm_points(x, alpha, num_points=20):
    """Get points along the arm for a given configuration"""
    arm_x = np.linspace(x, x + ARM_LENGTH * np.cos(alpha), num_points)
    arm_y = np.linspace(0, ARM_LENGTH * np.sin(alpha), num_points)
    
    # Add points for the width of the arm
    width_dir_x = -np.sin(alpha) * ARM_WIDTH/2
    width_dir_y = np.cos(alpha) * ARM_WIDTH/2
    
    arm_points = []
    for i in range(num_points):
        arm_points.append((arm_x[i] - width_dir_x, arm_y[i] - width_dir_y))
        arm_points.append((arm_x[i] + width_dir_x, arm_y[i] + width_dir_y))
    
    return np.array(arm_points)

def check_collision(x, alpha):
    """Check if the arm collides with any obstacle"""
    arm_points = get_arm_points(x, alpha)
    
    # Check collisions with obstacles
    for obstacle in OBSTACLES:
        distances = np.sqrt((arm_points[:, 0] - obstacle['x'])**2 + 
                           (arm_points[:, 1] - obstacle['y'])**2)
        if np.any(distances <= obstacle['radius']):
            return True
    
    return False

# Build configuration space
print("Building configuration space...")
for i, alpha in enumerate(alpha_values):
    for j, x in enumerate(x_values):
        if check_collision(x, alpha):
            config_space[i, j] = 1  # Mark as obstacle

print("Configuration space built!")

# Define start and end configurations
# We'll set these to configurations where the arm can reach the bins
def find_bin_configurations(bin_data):
    """Find possible configurations to reach the center of the bin"""
    bin_center_x = bin_data['x'] + bin_data['width'] / 2
    bin_center_y = bin_data['y'] + bin_data['height'] / 2
    
    valid_configs = []
    
    for i, alpha in enumerate(alpha_values):
        for j, x in enumerate(x_values):
            arm_end_x = x + ARM_LENGTH * np.cos(alpha)
            arm_end_y = ARM_LENGTH * np.sin(alpha)
            
            # Check if arm end is close to bin center
            distance = np.sqrt((arm_end_x - bin_center_x)**2 + (arm_end_y - bin_center_y)**2)
            
            if distance < 0.3 and not check_collision(x, alpha):  # Within 0.3 units of bin center
                valid_configs.append((i, j, distance))
    
    # Sort by distance and take the closest one
    if valid_configs:
        valid_configs.sort(key=lambda c: c[2])
        return valid_configs[0][0], valid_configs[0][1]
    
    # If no exact match found, find the closest possible configuration
    min_distance = float('inf')
    best_config = None
    
    for i, alpha in enumerate(alpha_values):
        for j, x in enumerate(x_values):
            if config_space[i, j] == 0:  # if not in collision
                arm_end_x = x + ARM_LENGTH * np.cos(alpha)
                arm_end_y = ARM_LENGTH * np.sin(alpha)
                
                distance = np.sqrt((arm_end_x - bin_center_x)**2 + (arm_end_y - bin_center_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_config = (i, j)
    
    return best_config

# Find start and end configurations
start_alpha_idx, start_x_idx = find_bin_configurations(RED_BIN)
end_alpha_idx, end_x_idx = find_bin_configurations(BLUE_BIN)

start_config = (start_alpha_idx, start_x_idx)
end_config = (end_alpha_idx, end_x_idx)

# Get the actual values
start_alpha = alpha_values[start_alpha_idx]
start_x = x_values[start_x_idx]
end_alpha = alpha_values[end_alpha_idx]
end_x = x_values[end_x_idx]

print(f"Start configuration: x={start_x:.2f}, alpha={start_alpha:.2f} rad")
print(f"End configuration: x={end_x:.2f}, alpha={end_alpha:.2f} rad")

# Path planning using A* algorithm
def heuristic(node, goal):
    """Calculate heuristic (Euclidean distance)"""
    return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

def get_neighbors(node, config_space):
    """Get valid neighboring configurations"""
    i, j = node
    neighbors = []
    
    # Check all 8 neighboring cells
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
                
            ni, nj = i + di, j + dj
            
            if (0 <= ni < config_space.shape[0] and 
                0 <= nj < config_space.shape[1] and 
                config_space[ni, nj] == 0):
                neighbors.append((ni, nj))
    
    return neighbors

def a_star(start, goal, config_space):
    """A* path planning algorithm"""
    open_set = {start}
    closed_set = set()
    
    # For node n, g_score[n] is the cost from start to n
    g_score = {start: 0}
    
    # For node n, f_score[n] = g_score[n] + heuristic(n, goal)
    f_score = {start: heuristic(start, goal)}
    
    # For node n, came_from[n] is the node immediately preceding it on the path
    came_from = {}
    
    while open_set:
        # Find node in open_set with the lowest f_score
        current = min(open_set, key=lambda n: f_score.get(n, float('inf')))
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]  # Return reversed path
        
        open_set.remove(current)
        closed_set.add(current)
        
        for neighbor in get_neighbors(current, config_space):
            if neighbor in closed_set:
                continue
            
            # Distance between current and neighbor
            # Diagonal movement costs sqrt(2)
            if abs(neighbor[0] - current[0]) == 1 and abs(neighbor[1] - current[1]) == 1:
                movement_cost = 1.414  # sqrt(2)
            else:
                movement_cost = 1
                
            tentative_g_score = g_score[current] + movement_cost
            
            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue
            
            # This path to neighbor is better than any previous one
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
    
    # No path found
    return None

# Find path in configuration space
print("Finding path...")
path_indices = a_star(start_config, end_config, config_space)

if path_indices:
    print(f"Path found with {len(path_indices)} steps!")
    
    # Convert indices to actual configuration values
    path = []
    for alpha_idx, x_idx in path_indices:
        path.append((alpha_values[alpha_idx], x_values[x_idx]))
    
    # Plot configuration space and path
    plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    
    # Plot configuration space
    ax1 = plt.subplot(gs[0])
    ax1.imshow(config_space, origin='lower', extent=[X_MIN, X_MAX, ALPHA_MIN, ALPHA_MAX], 
               aspect='auto', cmap='gray_r')
    
    # Plot path in configuration space
    if path:
        path_alpha = [p[0] for p in path]
        path_x = [p[1] for p in path]
        ax1.plot(path_x, path_alpha, 'r-', linewidth=2)
        ax1.plot(path_x[0], path_alpha[0], 'go', markersize=8)  # Start
        ax1.plot(path_x[-1], path_alpha[-1], 'bo', markersize=8)  # End
    
    ax1.set_xlabel('x position')
    ax1.set_ylabel('α angle (rad)')
    ax1.set_title('Configuration Space with Path')
    
    # Create workspace visualization
    ax2 = plt.subplot(gs[1])
    
    def update(frame):
        ax2.clear()
        
        # Set workspace limits
        ax2.set_xlim(0, WORKSPACE_WIDTH)
        ax2.set_ylim(0, WORKSPACE_HEIGHT)
        
        # Draw workspace boundaries
        ax2.plot([0, WORKSPACE_WIDTH], [0, 0], 'k-', linewidth=2)
        
        # Draw bins
        red_rect = Rectangle((RED_BIN['x'], RED_BIN['y']), RED_BIN['width'], RED_BIN['height'], 
                             color='red', alpha=0.7)
        blue_rect = Rectangle((BLUE_BIN['x'], BLUE_BIN['y']), BLUE_BIN['width'], BLUE_BIN['height'], 
                              color='blue', alpha=0.7)
        ax2.add_patch(red_rect)
        ax2.add_patch(blue_rect)
        
        # Draw obstacles
        for obstacle in OBSTACLES:
            circle = Circle((obstacle['x'], obstacle['y']), obstacle['radius'], color='gray')
            ax2.add_patch(circle)
        
        # Get current configuration
        if frame < len(path):
            alpha, x = path[frame]
        else:
            alpha, x = path[-1]
        
        # Draw robot arm
        arm_end_x = x + ARM_LENGTH * np.cos(alpha)
        arm_end_y = ARM_LENGTH * np.sin(alpha)
        
        # Draw the arm base
        base_width = 0.5
        base_height = 0.3
        base_rect = Rectangle((x - base_width/2, -base_height), base_width, base_height, 
                             color='lightgray', alpha=0.7)
        ax2.add_patch(base_rect)
        
        # Draw the arm with width
        width_dir_x = -np.sin(alpha) * ARM_WIDTH/2
        width_dir_y = np.cos(alpha) * ARM_WIDTH/2
        
        arm_points = [
            [x - width_dir_x, -width_dir_y],
            [x + width_dir_x, width_dir_y],
            [arm_end_x + width_dir_x, arm_end_y + width_dir_y],
            [arm_end_x - width_dir_x, arm_end_y - width_dir_y],
        ]
        
        arm_poly = plt.Polygon(arm_points, color='cyan', alpha=0.7)
        ax2.add_patch(arm_poly)
        
        # Draw gripper at the end
        gripper_size = 0.2
        gripper_circle = Circle((arm_end_x, arm_end_y), gripper_size, color='blue')
        ax2.add_patch(gripper_circle)
        
        # Draw coordinate system
        ax2.arrow(0, 0, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax2.text(1.1, 0, 'x', fontsize=12)
        
        # Add frame info
        ax2.set_title(f'Robot Workspace: Frame {frame}/{len(path)-1}')
        ax2.set_xlabel('x position')
        ax2.set_ylabel('y position')
        
        return []
    
    # Create animation
    num_frames = len(path)
    ani = FuncAnimation(plt.gcf(), update, frames=num_frames, 
                        interval=100, blit=True, repeat=True)
    
    plt.tight_layout()
    plt.savefig('robot_arm_path_planning.png')
    plt.show()
    
    # Save a few frames to demonstrate the path
    frames_to_save = [0, len(path)//4, len(path)//2, 3*len(path)//4, len(path)-1]
    
    fig, axs = plt.subplots(1, len(frames_to_save), figsize=(15, 3))
    
    for i, frame_idx in enumerate(frames_to_save):
        ax = axs[i]
        
        # Set workspace limits
        ax.set_xlim(0, WORKSPACE_WIDTH)
        ax.set_ylim(0, WORKSPACE_HEIGHT)
        
        # Draw workspace boundaries
        ax.plot([0, WORKSPACE_WIDTH], [0, 0], 'k-', linewidth=2)
        
        # Draw bins
        red_rect = Rectangle((RED_BIN['x'], RED_BIN['y']), RED_BIN['width'], RED_BIN['height'], 
                           color='red', alpha=0.7)
        blue_rect = Rectangle((BLUE_BIN['x'], BLUE_BIN['y']), BLUE_BIN['width'], BLUE_BIN['height'], 
                            color='blue', alpha=0.7)
        ax.add_patch(red_rect)
        ax.add_patch(blue_rect)
        
        # Draw obstacles
        for obstacle in OBSTACLES:
            circle = Circle((obstacle['x'], obstacle['y']), obstacle['radius'], color='gray')
            ax.add_patch(circle)
        
        # Get current configuration
        alpha, x = path[frame_idx]
        
        # Draw robot arm
        arm_end_x = x + ARM_LENGTH * np.cos(alpha)
        arm_end_y = ARM_LENGTH * np.sin(alpha)
        
        # Draw the arm base
        base_width = 0.5
        base_height = 0.3
        base_rect = Rectangle((x - base_width/2, -base_height), base_width, base_height, 
                             color='lightgray', alpha=0.7)
        ax.add_patch(base_rect)
        
        # Draw the arm with width
        width_dir_x = -np.sin(alpha) * ARM_WIDTH/2
        width_dir_y = np.cos(alpha) * ARM_WIDTH/2
        
        arm_points = [
            [x - width_dir_x, -width_dir_y],
            [x + width_dir_x, width_dir_y],
            [arm_end_x + width_dir_x, arm_end_y + width_dir_y],
            [arm_end_x - width_dir_x, arm_end_y - width_dir_y],
        ]
        
        arm_poly = plt.Polygon(arm_points, color='cyan', alpha=0.7)
        ax.add_patch(arm_poly)
        
        # Draw gripper at the end
        gripper_size = 0.2
        gripper_circle = Circle((arm_end_x, arm_end_y), gripper_size, color='blue')
        ax.add_patch(gripper_circle)
        
        ax.set_title(f'Step {frame_idx}')
        
        # Only add x/y labels for leftmost plot
        if i == 0:
            ax.set_xlabel('x position')
            ax.set_ylabel('y position')
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('robot_arm_path_sequence.png')
    plt.show()
    
    print("Visualization complete!")
else:
    print("No path found. Try adjusting the workspace dimensions or obstacle positions.")