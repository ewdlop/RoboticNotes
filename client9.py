import pybullet as p
import time
import pybullet_data
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import json
import os

class RobotController:
    def __init__(self):
        self.physicsClient = None
        self.robotId = None
        self.planeId = None
        self.cubeId = None
        self.controllable_joints = []
        self.joint_info = {}
        self.joint_patterns = {}
        
        # Locomotion parameters for each joint: [offset, amplitude, phase, frequency]
        self.locomotion_params = {}
        
        # Cube following parameters
        self.cube_position = [3, 0, 0.5]  # Fixed cube position (immovable)
        
        # Learning tracking
        self.learning_history = []
        self.best_fitness = -float('inf')
        self.best_params = None
        
        # File for saving/loading parameters
        self.params_file = "best_cube_following_params.json"
        
    def setup_simulation(self):
        """Initialize PyBullet simulation environment"""
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")
        
        # Create red cube target
        self.create_target_cube()
        
    def create_target_cube(self):
        """Create a red cube as the target"""
        # Create cube collision shape
        cube_size = 0.2
        cube_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size/2, cube_size/2, cube_size/2])
        cube_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[cube_size/2, cube_size/2, cube_size/2], 
                                               rgbaColor=[1, 0, 0, 1])  # Red color
        
        # Create cube body (static and non-moveable)
        self.cubeId = p.createMultiBody(baseMass=0,  # Mass = 0 makes it static
                                       baseCollisionShapeIndex=cube_collision_shape,
                                       baseVisualShapeIndex=cube_visual_shape,
                                       basePosition=self.cube_position)
        
        # Make cube completely immovable by fixing it in place
        p.createConstraint(self.cubeId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], self.cube_position)
        
        print(f"Red target cube created (immovable) at position: {self.cube_position}")
        
    def update_cube_position(self, simulation_time):
        """Update cube position based on movement pattern"""
        if self.cube_movement_type == 'circular':
            # Circular movement
            center_x, center_y = 0, 0
            angle = simulation_time * self.cube_speed
            new_x = center_x + self.cube_path_radius * math.cos(angle)
            new_y = center_y + self.cube_path_radius * math.sin(angle)
            new_z = 0.5
            
        elif self.cube_movement_type == 'linear':
            # Linear back and forth movement
            amplitude = self.cube_path_radius
            new_x = amplitude * math.sin(simulation_time * self.cube_speed)
            new_y = 0
            new_z = 0.5
            
        elif self.cube_movement_type == 'random':
            # Random movement with smooth transitions
            if not hasattr(self, 'cube_target'):
                self.cube_target = [random.uniform(-3, 3), random.uniform(-3, 3), 0.5]
                self.cube_change_time = simulation_time + random.uniform(2, 5)
            
            if simulation_time > self.cube_change_time:
                self.cube_target = [random.uniform(-3, 3), random.uniform(-3, 3), 0.5]
                self.cube_change_time = simulation_time + random.uniform(2, 5)
            
            # Smooth interpolation towards target
            current_pos = self.cube_position
            new_x = current_pos[0] + (self.cube_target[0] - current_pos[0]) * 0.02
            new_y = current_pos[1] + (self.cube_target[1] - current_pos[1]) * 0.02
            new_z = 0.5
        
        self.cube_position = [new_x, new_y, new_z]
        p.resetBasePositionAndOrientation(self.cubeId, self.cube_position, [0, 0, 0, 1])
        
    def load_robot(self):
        """Load the ant robot from MJCF file"""
        try:
            robotIds = p.loadMJCF("ant.xml")
            print(f"MJCF loaded successfully. Number of bodies: {len(robotIds)}")
            
            if len(robotIds) > 1:
                self.robotId = robotIds[1]
            else:
                self.robotId = robotIds[0]
                
            print(f"Using robotId: {self.robotId}")
            
            # Set initial position
            p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])
            
            basePos, baseOrn = p.getBasePositionAndOrientation(self.robotId)
            print(f"Robot base position: {basePos}")
            return True
            
        except Exception as e:
            print(f"Failed to load MJCF: {e}")
            return False

    def analyze_joints(self):
        """Analyze and categorize all joints in the robot"""
        numJoints = p.getNumJoints(self.robotId)
        print(f"\nNumber of joints in the robot: {numJoints}")
        
        self.controllable_joints = []
        self.joint_info = {}
        
        for i in range(numJoints):
            jointInfo = p.getJointInfo(self.robotId, i)
            jointName = jointInfo[1].decode('utf-8')
            jointType = jointInfo[2]
            jointLowerLimit = jointInfo[8]
            jointUpperLimit = jointInfo[9]
            
            self.joint_info[i] = {
                'name': jointName,
                'type': jointType,
                'lower_limit': jointLowerLimit,
                'upper_limit': jointUpperLimit
            }
            
            # Skip the root joint (free joint) and include hinge joints
            if jointType == 0 and jointName != 'root':  # Hinge joints but not root
                if jointLowerLimit < jointUpperLimit:
                    self.controllable_joints.append(i)
                    print(f"Controllable joint {i}: {jointName}")
        
        print(f"Total controllable joints: {len(self.controllable_joints)}")
        self.create_joint_patterns()
        
    def create_joint_patterns(self):
        """Create control patterns for different joint groups"""
        self.joint_patterns = {}
        
        for joint_idx in self.controllable_joints:
            joint_name = self.joint_info[joint_idx]['name'].lower()
            
            if 'hip' in joint_name:
                self.joint_patterns[joint_idx] = 'hip'
            elif 'ankle' in joint_name:
                self.joint_patterns[joint_idx] = 'ankle'
            else:
                self.joint_patterns[joint_idx] = 'other'
    
    def initialize_random_params(self):
        """Initialize random locomotion parameters for each joint"""
        self.locomotion_params = {}
        
        for joint_idx in self.controllable_joints:
            joint_name = self.joint_info[joint_idx]['name']
            lower_limit = self.joint_info[joint_idx]['lower_limit']
            upper_limit = self.joint_info[joint_idx]['upper_limit']
            
            # Random parameters: [offset, amplitude, phase, frequency]
            range_size = upper_limit - lower_limit
            offset = random.uniform(lower_limit + 0.1 * range_size, 
                                  upper_limit - 0.1 * range_size)
            amplitude = random.uniform(0.1 * range_size, 0.4 * range_size)
            phase = random.uniform(0, 2 * math.pi)
            frequency = random.uniform(0.5, 2.0)  # Hz
            
            self.locomotion_params[joint_idx] = [offset, amplitude, phase, frequency]
            print(f"Joint {joint_idx} ({joint_name}): offset={offset:.3f}, amp={amplitude:.3f}, phase={phase:.3f}, freq={frequency:.3f}")

    def calculate_fitness(self, robot_positions, cube_positions, duration):
        """Calculate fitness based on cube following and speed"""
        if len(robot_positions) < 2 or len(cube_positions) < 2:
            return 0.0
        
        # Calculate average distance to cube
        distances = []
        speeds = []
        
        for i in range(len(robot_positions)):
            if i < len(cube_positions):
                # Distance to cube
                dist = math.sqrt(
                    (robot_positions[i][0] - cube_positions[i][0])**2 + 
                    (robot_positions[i][1] - cube_positions[i][1])**2
                )
                distances.append(dist)
        
        # Calculate robot speed
        for i in range(1, len(robot_positions)):
            speed = math.sqrt(
                (robot_positions[i][0] - robot_positions[i-1][0])**2 + 
                (robot_positions[i][1] - robot_positions[i-1][1])**2
            ) * 240  # Convert to m/s (240 Hz simulation)
            speeds.append(speed)
        
        avg_distance = np.mean(distances) if distances else float('inf')
        avg_speed = np.mean(speeds) if speeds else 0.0
        
        # Fitness function: reward speed and penalize distance from cube
        # Higher speed is better, lower distance is better
        distance_penalty = max(0, avg_distance - 0.5)  # Only penalize if > 0.5m away
        fitness = avg_speed - 2.0 * distance_penalty
        
        return max(0, fitness)  # Ensure non-negative fitness

    def evaluate_locomotion(self, params, duration=8.0):
        """Evaluate locomotion performance with cube following"""
        # Reset robot position
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])
        
        # Let robot stabilize
        for _ in range(100):
            p.stepSimulation()
        
        # Track positions
        robot_positions = []
        
        # Run simulation
        steps = int(duration * 240)  # 240 Hz simulation
        
        for step in range(steps):
            simulation_time = step * (1.0 / 240.0)
            
            # Get current positions
            robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)
            cube_pos = self.cube_position  # Fixed position
            
            # Store positions every 10 steps to reduce memory usage
            if step % 10 == 0:
                robot_positions.append(robot_pos)
            
            # Calculate direction to cube for adaptive control
            dx = cube_pos[0] - robot_pos[0]
            dy = cube_pos[1] - robot_pos[1]
            distance_to_cube = math.sqrt(dx**2 + dy**2)
            
            # Normalize direction
            if distance_to_cube > 0:
                direction_x = dx / distance_to_cube
                direction_y = dy / distance_to_cube
            else:
                direction_x = direction_y = 0
            
            # Apply control to each joint with directional bias
            for joint_idx in self.controllable_joints:
                offset, amplitude, phase, frequency = params[joint_idx]
                lower_limit = self.joint_info[joint_idx]['lower_limit']
                upper_limit = self.joint_info[joint_idx]['upper_limit']
                
                # Basic sinusoidal motion with frequency
                base_motion = offset + amplitude * math.sin(2 * math.pi * frequency * simulation_time + phase)
                
                # Add directional bias based on joint type and cube direction
                joint_name = self.joint_info[joint_idx]['name'].lower()
                bias = 0
                
                if 'hip' in joint_name and distance_to_cube > 0.5:
                    # Add directional bias for hip joints
                    if 'front' in joint_name or '0' in joint_name or '1' in joint_name:
                        bias = 0.1 * direction_x * (distance_to_cube / 5.0)
                    elif 'back' in joint_name or '2' in joint_name or '3' in joint_name:
                        bias = -0.1 * direction_x * (distance_to_cube / 5.0)
                
                target = base_motion + bias
                
                # Clamp to joint limits
                target = max(lower_limit, min(upper_limit, target))
                
                # Apply control
                p.setJointMotorControl2(
                    bodyUniqueId=self.robotId,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=200,  # Increased force for better following
                    maxVelocity=3.0
                )
            
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
        
        # Calculate fitness: distance to cube + speed
        final_robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        
        # Distance to cube (lower is better)
        final_distance = math.sqrt(
            (final_robot_pos[0] - cube_pos[0])**2 + 
            (final_robot_pos[1] - cube_pos[1])**2
        )
        
        # Speed calculation (higher is better)
        if len(robot_positions) >= 2:
            total_distance = 0
            for i in range(1, len(robot_positions)):
                dist = math.sqrt(
                    (robot_positions[i][0] - robot_positions[i-1][0])**2 + 
                    (robot_positions[i][1] - robot_positions[i-1][1])**2
                )
                total_distance += dist
            avg_speed = total_distance / duration
        else:
            avg_speed = 0
        
        # Fitness: reward speed and getting close to cube
        distance_penalty = min(final_distance, 5.0)  # Cap penalty at 5m
        fitness = avg_speed - distance_penalty + (5.0 - final_distance) * 0.1
        
        return max(0, fitness)  # Ensure non-negative fitness
    
    def mutate_params(self, params, mutation_rate=0.15):
        """Create a mutated version of parameters"""
        new_params = {}
        
        for joint_idx in params:
            new_params[joint_idx] = params[joint_idx].copy()
            lower_limit = self.joint_info[joint_idx]['lower_limit']
            upper_limit = self.joint_info[joint_idx]['upper_limit']
            range_size = upper_limit - lower_limit
            
            for i in range(4):  # offset, amplitude, phase, frequency
                if random.random() < mutation_rate:
                    if i == 0:  # offset
                        new_params[joint_idx][i] += random.gauss(0, 0.1 * range_size)
                        new_params[joint_idx][i] = max(lower_limit, 
                                                     min(upper_limit, new_params[joint_idx][i]))
                    elif i == 1:  # amplitude
                        new_params[joint_idx][i] += random.gauss(0, 0.05 * range_size)
                        new_params[joint_idx][i] = max(0, 
                                                     min(0.5 * range_size, new_params[joint_idx][i]))
                    elif i == 2:  # phase
                        new_params[joint_idx][i] += random.gauss(0, 0.3)
                        new_params[joint_idx][i] = new_params[joint_idx][i] % (2 * math.pi)
                    else:  # frequency
                        new_params[joint_idx][i] += random.gauss(0, 0.2)
                        new_params[joint_idx][i] = max(0.2, min(3.0, new_params[joint_idx][i]))
        
        return new_params
    
    def save_best_params(self):
        """Save the best parameters to a JSON file"""
        if self.best_params is None:
            print("No best parameters to save!")
            return False
            
        try:
            # Convert joint indices to strings for JSON compatibility
            params_data = {
                'best_fitness': self.best_fitness,
                'best_params': {str(k): v for k, v in self.best_params.items()},
                'joint_info': {str(k): v for k, v in self.joint_info.items()},
                'learning_history': self.learning_history,
                'cube_position': self.cube_position
            }
            
            with open(self.params_file, 'w') as f:
                json.dump(params_data, f, indent=2)
            
            print(f"Best parameters saved to {self.params_file}")
            print(f"Best fitness achieved: {self.best_fitness:.4f}")
            return True
            
        except Exception as e:
            print(f"Error saving parameters: {e}")
            return False
    
    def load_best_params(self):
        """Load the best parameters from a JSON file"""
        if not os.path.exists(self.params_file):
            print(f"No saved parameters found at {self.params_file}")
            return False
            
        try:
            with open(self.params_file, 'r') as f:
                params_data = json.load(f)
            
            # Convert string keys back to integers
            self.best_fitness = params_data['best_fitness']
            self.best_params = {int(k): v for k, v in params_data['best_params'].items()}
            self.learning_history = params_data.get('learning_history', [])
            self.cube_position = params_data.get('cube_position', [3, 0, 0.5])
            
            print(f"Best parameters loaded from {self.params_file}")
            print(f"Loaded best fitness: {self.best_fitness:.4f}")
            
            # Verify that loaded joints match current robot
            if self.joint_info:
                loaded_joints = set(self.best_params.keys())
                current_joints = set(self.controllable_joints)
                
                if loaded_joints != current_joints:
                    print("Warning: Loaded parameters don't match current robot joints!")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error loading parameters: {e}")
            return False
    
    def hill_climber_optimization(self, generations=50):
        """Optimize locomotion parameters using hill climbing for cube following"""
        print(f"\nStarting cube-following optimization for {generations} generations...")
        print(f"Target cube at fixed position: {self.cube_position}")
        
        self.learning_history = []
        
        # Initialize with random parameters
        self.initialize_random_params()
        current_params = self.locomotion_params.copy()
        current_fitness = self.evaluate_locomotion(current_params)
        
        self.best_fitness = current_fitness
        self.best_params = current_params.copy()
        self.learning_history.append(current_fitness)
        
        print(f"Initial fitness: {current_fitness:.4f}")
        
        for generation in range(generations):
            # Create mutated version
            candidate_params = self.mutate_params(current_params)
            candidate_fitness = self.evaluate_locomotion(candidate_params)
            
            # Accept if better
            if candidate_fitness > current_fitness:
                current_params = candidate_params
                current_fitness = candidate_fitness
                print(f"Generation {generation+1}: NEW BEST fitness = {current_fitness:.4f}")
                
                if current_fitness > self.best_fitness:
                    self.best_fitness = current_fitness
                    self.best_params = current_params.copy()
            else:
                print(f"Generation {generation+1}: fitness = {candidate_fitness:.4f} (no improvement)")
            
            self.learning_history.append(current_fitness)
        
        print(f"\nOptimization complete!")
        print(f"Best fitness achieved: {self.best_fitness:.4f}")
        
        # Automatically save the best parameters
        self.save_best_params()
        
        return self.best_params
    
    def plot_learning_curve(self):
        """Plot the learning curve"""
        if not self.learning_history:
            print("No learning history to plot!")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Main learning curve
        plt.subplot(2, 1, 1)
        plt.plot(self.learning_history, linewidth=2, color='blue')
        plt.title('Robot Cube-Following Learning Curve')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.grid(True, alpha=0.3)
        
        # Moving average for trend
        if len(self.learning_history) > 10:
            window = min(10, len(self.learning_history) // 4)
            moving_avg = np.convolve(self.learning_history, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(self.learning_history)), moving_avg, 
                    linewidth=2, color='red', alpha=0.7, label=f'Moving Avg ({window})')
            plt.legend()
        
        # Improvement histogram
        plt.subplot(2, 1, 2)
        improvements = [self.learning_history[i] - self.learning_history[i-1] 
                       for i in range(1, len(self.learning_history))]
        plt.hist(improvements, bins=20, alpha=0.7, color='green')
        plt.title('Distribution of Fitness Improvements')
        plt.xlabel('Fitness Change')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nLearning Statistics:")
        print(f"Initial fitness: {self.learning_history[0]:.4f}")
        print(f"Final fitness: {self.learning_history[-1]:.4f}")
        print(f"Best fitness: {max(self.learning_history):.4f}")
        print(f"Total improvement: {self.learning_history[-1] - self.learning_history[0]:.4f}")
        print(f"Success rate: {sum(1 for x in improvements if x > 0) / len(improvements) * 100:.1f}%")
    
    def demonstrate_best_params(self, duration=15.0):
        """Demonstrate the robot using the best found parameters"""
        if self.best_params is None:
            print("No optimized parameters found! Run optimization first.")
            return
            
        print(f"\nDemonstrating cube following for {duration} seconds...")
        print(f"Target cube at fixed position: {self.cube_position}")
        
        # Reset robot
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])
        
        steps = int(duration * 240)
        robot_positions = []
        
        for step in range(steps):
            simulation_time = step * (1.0 / 240.0)
            
            # Get current positions
            robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)
            cube_pos = self.cube_position  # Fixed position
            
            # Store positions for analysis
            if step % 24 == 0:  # Every 0.1 seconds
                robot_positions.append(robot_pos)
            
            # Calculate direction to cube
            dx = cube_pos[0] - robot_pos[0]
            dy = cube_pos[1] - robot_pos[1]
            distance_to_cube = math.sqrt(dx**2 + dy**2)
            
            if distance_to_cube > 0:
                direction_x = dx / distance_to_cube
                direction_y = dy / distance_to_cube
            else:
                direction_x = direction_y = 0
            
            # Apply best parameters with directional bias
            for joint_idx in self.controllable_joints:
                offset, amplitude, phase, frequency = self.best_params[joint_idx]
                lower_limit = self.joint_info[joint_idx]['lower_limit']
                upper_limit = self.joint_info[joint_idx]['upper_limit']
                
                base_motion = offset + amplitude * math.sin(2 * math.pi * frequency * simulation_time + phase)
                
                # Add directional bias
                joint_name = self.joint_info[joint_idx]['name'].lower()
                bias = 0
                
                if 'hip' in joint_name and distance_to_cube > 0.5:
                    if 'front' in joint_name or '0' in joint_name or '1' in joint_name:
                        bias = 0.1 * direction_x * (distance_to_cube / 5.0)
                    elif 'back' in joint_name or '2' in joint_name or '3' in joint_name:
                        bias = -0.1 * direction_x * (distance_to_cube / 5.0)
                
                target = base_motion + bias
                target = max(lower_limit, min(upper_limit, target))
                
                p.setJointMotorControl2(
                    bodyUniqueId=self.robotId,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=200,
                    maxVelocity=3.0
                )
            
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
            
            # Print status every 3 seconds
            if step % 720 == 0:
                print(f"Time {step/240:.1f}s: Robot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), "
                      f"Distance to cube: {distance_to_cube:.2f}m")
        
        # Calculate final performance
        final_robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        final_distance = math.sqrt(
            (final_robot_pos[0] - cube_pos[0])**2 + 
            (final_robot_pos[1] - cube_pos[1])**2
        )
        print(f"\nFinal distance to cube: {final_distance:.2f}m")

def main():
    """Main function to run the complete simulation"""
    controller = RobotController()
    
    # Setup simulation
    print("Setting up simulation with red cube target...")
    controller.setup_simulation()
    
    # Load robot
    if not controller.load_robot():
        print("Failed to load robot!")
        return
    
    # Analyze joints
    controller.analyze_joints()
    
    if not controller.controllable_joints:
        print("No controllable joints found!")
        return
    
    # Set cube position
    print(f"\nRed cube is fixed at position: {controller.cube_position}")
    
    try:
        print("\nChoose option:")
        print("1. Run new optimization")
        print("2. Load saved parameters and demonstrate")
        print("3. Load saved parameters and continue optimization")
        
        choice = input("Enter choice (1, 2, or 3): ").strip()
        
        if choice == "2":
            # Load and demonstrate
            if controller.load_best_params():
                print("Demonstrating loaded parameters...")
                controller.demonstrate_best_params(duration=20.0)
            else:
                print("Failed to load parameters. Running new optimization instead.")
                choice = "1"
        
        elif choice == "3":
            # Load and continue optimization
            if controller.load_best_params():
                print("Continuing optimization from loaded parameters...")
                controller.locomotion_params = controller.best_params.copy()
                generations = int(input("Enter number of additional generations (default 25): ") or "25")
                controller.hill_climber_optimization(generations=generations)
                controller.plot_learning_curve()
                
                print(f"\nPress Enter to see demonstration of optimized cube-following robot...")
                input()
                controller.demonstrate_best_params(duration=20.0)
            else:
                print("Failed to load parameters. Running new optimization instead.")
                choice = "1"
        
        if choice == "1":
            # Run new optimization
            print("Starting new cube-following optimization...")
            generations = int(input("Enter number of generations (default 40): ") or "40")
            best_params = controller.hill_climber_optimization(generations=generations)
            
            # Plot learning curve
            controller.plot_learning_curve()
            
            # Demonstrate best solution
            print(f"\nPress Enter to see demonstration of optimized cube-following robot...")
            input()
            controller.demonstrate_best_params(duration=20.0)
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Disconnecting...")
        p.disconnect()

if __name__ == "__main__":
    main()