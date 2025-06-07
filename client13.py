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
        self.controllable_joints = []
        self.joint_info = {}
        self.joint_patterns = {}
        
        # Locomotion parameters for each joint: [offset, amplitude, phase]
        self.locomotion_params = {}
        
        # Learning tracking
        self.learning_history = []
        self.best_speed = -float('inf')
        self.best_params = None
        
        # File for saving/loading parameters
        self.params_file = "best_robot_params.json"
        
    def setup_simulation(self):
        """Initialize PyBullet simulation environment"""
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")
        
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
            
            # Random parameters: [offset, amplitude, phase]
            range_size = upper_limit - lower_limit
            offset = random.uniform(lower_limit + 0.1 * range_size, 
                                  upper_limit - 0.1 * range_size)
            amplitude = random.uniform(0.1 * range_size, 0.4 * range_size)
            phase = random.uniform(0, 2 * math.pi)
            
            self.locomotion_params[joint_idx] = [offset, amplitude, phase]
            print(f"Joint {joint_idx} ({joint_name}): offset={offset:.3f}, amp={amplitude:.3f}, phase={phase:.3f}")
    


    def evaluate_locomotion(self, params, duration=5.0):
        """Evaluate locomotion performance with given parameters"""
        # Reset robot position
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])
        
        # Let robot stabilize
        for _ in range(100):
            p.stepSimulation()
        
        # Record initial position
        initial_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        
        # Run simulation
        steps = int(duration * 240)  # 240 Hz simulation
        
        for step in range(steps):
            time_factor = step * (1.0 / 240.0)
            
            # Apply control to each joint
            for joint_idx in self.controllable_joints:
                offset, amplitude, phase = params[joint_idx]
                lower_limit = self.joint_info[joint_idx]['lower_limit']
                upper_limit = self.joint_info[joint_idx]['upper_limit']
                
                # Basic sinusoidal motion
                target = offset + amplitude * math.sin(2 * math.pi * time_factor + phase)
                
                # Clamp to joint limits
                target = max(lower_limit, min(upper_limit, target))
                
                # Apply control
                p.setJointMotorControl2(
                    bodyUniqueId=self.robotId,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=150,
                    maxVelocity=2.0
                )
            
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
        
        # Calculate final distance traveled
        final_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        
        # Calculate forward speed
        distance_traveled = math.sqrt(
            (final_pos[0] - initial_pos[0])**2 + 
            (final_pos[1] - initial_pos[1])**2
        )
        fitness = distance_traveled / duration  # Speed
        
        return fitness
    
    def mutate_params(self, params, mutation_rate=0.1):
        """Create a mutated version of parameters"""
        new_params = {}
        
        for joint_idx in params:
            new_params[joint_idx] = params[joint_idx].copy()
            lower_limit = self.joint_info[joint_idx]['lower_limit']
            upper_limit = self.joint_info[joint_idx]['upper_limit']
            range_size = upper_limit - lower_limit
            
            for i in range(3):  # offset, amplitude, phase
                if random.random() < mutation_rate:
                    if i == 0:  # offset
                        new_params[joint_idx][i] += random.gauss(0, 0.1 * range_size)
                        new_params[joint_idx][i] = max(lower_limit, 
                                                     min(upper_limit, new_params[joint_idx][i]))
                    elif i == 1:  # amplitude
                        new_params[joint_idx][i] += random.gauss(0, 0.05 * range_size)
                        new_params[joint_idx][i] = max(0, 
                                                     min(0.5 * range_size, new_params[joint_idx][i]))
                    else:  # phase
                        new_params[joint_idx][i] += random.gauss(0, 0.3)
                        new_params[joint_idx][i] = new_params[joint_idx][i] % (2 * math.pi)
        
        return new_params
    
    def save_best_params(self):
        """Save the best parameters to a JSON file"""
        if self.best_params is None:
            print("No best parameters to save!")
            return False
            
        try:
            # Convert joint indices to strings for JSON compatibility
            params_data = {
                'best_speed': self.best_speed,
                'best_params': {str(k): v for k, v in self.best_params.items()},
                'joint_info': {str(k): v for k, v in self.joint_info.items()},
                'learning_history': self.learning_history
            }
            
            with open(self.params_file, 'w') as f:
                json.dump(params_data, f, indent=2)
            
            print(f"Best parameters saved to {self.params_file}")
            print(f"Best speed achieved: {self.best_speed:.4f} m/s")
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
            self.best_speed = params_data['best_speed']
            self.best_params = {int(k): v for k, v in params_data['best_params'].items()}
            self.learning_history = params_data.get('learning_history', [])
            
            print(f"Best parameters loaded from {self.params_file}")
            print(f"Loaded best speed: {self.best_speed:.4f} m/s")
            
            # Verify that loaded joints match current robot
            if self.joint_info:
                loaded_joints = set(self.best_params.keys())
                current_joints = set(self.controllable_joints)
                
                if loaded_joints != current_joints:
                    print("Warning: Loaded parameters don't match current robot joints!")
                    print(f"Loaded joints: {loaded_joints}")
                    print(f"Current joints: {current_joints}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error loading parameters: {e}")
            return False
    
    def print_saved_params_info(self):
        """Print information about saved parameters without loading them"""
        if not os.path.exists(self.params_file):
            print(f"No saved parameters found at {self.params_file}")
            return
            
        try:
            with open(self.params_file, 'r') as f:
                params_data = json.load(f)
            
            print(f"\nSaved Parameters Info:")
            print(f"  File: {self.params_file}")
            print(f"  Best Speed: {params_data['best_speed']:.4f} m/s")
            print(f"  Number of joints: {len(params_data['best_params'])}")
            print(f"  Learning history length: {len(params_data.get('learning_history', []))}")
            
            if 'best_params' in params_data:
                print("  Joint parameters:")
                joint_info = params_data.get('joint_info', {})
                for joint_idx_str, params in params_data['best_params'].items():
                    joint_name = joint_info.get(joint_idx_str, {}).get('name', f'Joint_{joint_idx_str}')
                    print(f"    {joint_name}: offset={params[0]:.3f}, amp={params[1]:.3f}, phase={params[2]:.3f}")
                    
        except Exception as e:
            print(f"Error reading saved parameters: {e}")
    
    def hill_climber_optimization(self, generations=50):
        """Optimize locomotion parameters using hill climbing"""
        print(f"\nStarting hill climber optimization for {generations} generations...")
        
        self.learning_history = []
        
        # Initialize with random parameters
        self.initialize_random_params()
        current_params = self.locomotion_params.copy()
        current_fitness = self.evaluate_locomotion(current_params)
        
        self.best_speed = current_fitness
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
                
                if current_fitness > self.best_speed:
                    self.best_speed = current_fitness
                    self.best_params = current_params.copy()
            else:
                print(f"Generation {generation+1}: fitness = {candidate_fitness:.4f} (no improvement)")
            
            self.learning_history.append(current_fitness)
        
        print(f"\nOptimization complete!")
        print(f"Best fitness achieved: {self.best_speed:.4f}")
        
        # Automatically save the best parameters
        self.save_best_params()
        
        return self.best_params
    
    def plot_learning_curve(self):
        """Plot the learning curve"""
        if not self.learning_history:
            print("No learning history to plot!")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.learning_history, linewidth=2)
        plt.title('Robot Locomotion Learning Curve')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Speed m/s)')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Print statistics
        print(f"\nLearning Statistics:")
        print(f"Initial fitness: {self.learning_history[0]:.4f}")
        print(f"Final fitness: {self.learning_history[-1]:.4f}")
        print(f"Best fitness: {max(self.learning_history):.4f}")
        print(f"Improvement: {self.learning_history[-1] - self.learning_history[0]:.4f}")
    
    def demonstrate_best_params(self, duration=10.0):
        """Demonstrate the robot using the best found parameters"""
        if self.best_params is None:
            print("No optimized parameters found! Run optimization first.")
            return
            
        print(f"\nDemonstrating best parameters for {duration} seconds...")
        print("Best parameters:")
        for joint_idx, params in self.best_params.items():
            joint_name = self.joint_info[joint_idx]['name']
            print(f"  {joint_name}: offset={params[0]:.3f}, amp={params[1]:.3f}, phase={params[2]:.3f}")
        
        # Reset robot
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])
        
        steps = int(duration * 240)
        
        for step in range(steps):
            time_factor = step * (1.0 / 240.0)
            
            # Apply best parameters
            for joint_idx in self.controllable_joints:
                offset, amplitude, phase = self.best_params[joint_idx]
                lower_limit = self.joint_info[joint_idx]['lower_limit']
                upper_limit = self.joint_info[joint_idx]['upper_limit']
                
                target = offset + amplitude * math.sin(2 * math.pi * time_factor + phase)
                target = max(lower_limit, min(upper_limit, target))
                
                p.setJointMotorControl2(
                    bodyUniqueId=self.robotId,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=150,
                    maxVelocity=2.0
                )
            
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
            
            # Print status every 2 seconds
            if step % 480 == 0:
                robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)
                print(f"Time {step/240:.1f}s: Robot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")

def main():
    """Main function to run the complete simulation"""
    controller = RobotController()
    
    # Setup simulation
    print("Setting up simulation...")
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
    
    # Check for saved parameters
    controller.print_saved_params_info()
    
    print("\nChoose option:")
    print("1. Run new optimization")
    print("2. Load saved parameters and demonstrate")
    print("3. Load saved parameters and continue optimization")
    
    try:
        choice = input("Enter choice (1, 2, or 3): ").strip()
        
        if choice == "2":
            # Load and demonstrate
            if controller.load_best_params():
                print("Demonstrating loaded parameters...")
                controller.demonstrate_best_params(duration=15.0)
            else:
                print("Failed to load parameters. Running new optimization instead.")
                choice = "1"
        
        elif choice == "3":
            # Load and continue optimization
            if controller.load_best_params():
                print("Continuing optimization from loaded parameters...")
                # Set current params to loaded params
                controller.locomotion_params = controller.best_params.copy()
                generations = int(input("Enter number of additional generations (default 20): ") or "20")
                controller.hill_climber_optimization(generations=generations)
                controller.plot_learning_curve()
                
                print(f"\nPress Enter to see demonstration of optimized robot...")
                input()
                controller.demonstrate_best_params(duration=15.0)
            else:
                print("Failed to load parameters. Running new optimization instead.")
                choice = "1"
        
        if choice == "1":
            # Run new optimization
            print("Starting new optimization for forward locomotion...")
            generations = int(input("Enter number of generations (default 30): ") or "30")
            best_params = controller.hill_climber_optimization(generations=generations)
            
            # Plot learning curve
            controller.plot_learning_curve()
            
            # Demonstrate best solution
            print(f"\nPress Enter to see demonstration of optimized robot...")
            input()
            controller.demonstrate_best_params(duration=15.0)
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Disconnecting...")
        p.disconnect()

if __name__ == "__main__":
    main()