import pybullet as p
import time
import pybullet_data
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

class RobotController:
    def __init__(self):
        self.physicsClient = None
        self.robotId = None
        self.planeId = None
        self.red_cube_id = None
        self.controllable_joints = []
        self.joint_info = {}
        self.joint_patterns = {}
        
        # Locomotion parameters for each joint: [offset, amplitude, phase]
        self.locomotion_params = {}
        
        # Learning tracking
        self.learning_history = []
        self.best_speed = -float('inf')
        self.best_params = None
        
        # Vision parameters
        self.camera_width = 128
        self.camera_height = 96
        self.use_vision = False
        
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
    
    def add_red_cube(self):
        """Add a red cube target for vision-based chasing"""
        # Create a red cube at a random position
        cube_pos = [random.uniform(-5, 5), random.uniform(-5, 5), 0.5]
        cube_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # Create collision shape and visual shape
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3])
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3], 
                                         rgbaColor=[1, 0, 0, 1])  # Red color
        
        self.red_cube_id = p.createMultiBody(baseMass=1,
                                           baseCollisionShapeIndex=collision_shape,
                                           baseVisualShapeIndex=visual_shape,
                                           basePosition=cube_pos,
                                           baseOrientation=cube_orientation)
        print(f"Red cube added at position: {cube_pos}")
        
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
    
    def get_camera_image(self):
        """Get camera image from robot's perspective"""
        if not self.use_vision:
            return None
            
        # Get robot's position and orientation
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robotId)
        
        # Calculate camera position (slightly above and forward from robot center)
        camera_pos = [robot_pos[0], robot_pos[1], robot_pos[2] + 0.3]
        
        # Calculate target position (looking forward)
        euler_angles = p.getEulerFromQuaternion(robot_orn)
        yaw = euler_angles[2]
        target_pos = [
            camera_pos[0] + 2 * math.cos(yaw),
            camera_pos[1] + 2 * math.sin(yaw),
            camera_pos[2]
        ]
        
        # Camera up vector
        up_vector = [0, 0, 1]
        
        # Get view and projection matrices
        view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=self.camera_width/self.camera_height, 
            nearVal=0.1, farVal=10
        )
        
        # Render image
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.camera_width, height=self.camera_height,
            viewMatrix=view_matrix, projectionMatrix=projection_matrix
        )
        
        return rgb_img
    
    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV color space using numpy"""
        rgb = rgb.astype(np.float32) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val
        
        # Hue calculation
        h = np.zeros_like(max_val)
        mask = diff != 0
        
        # Red is max
        r_mask = mask & (max_val == r)
        h[r_mask] = (60 * ((g[r_mask] - b[r_mask]) / diff[r_mask]) + 360) % 360
        
        # Green is max
        g_mask = mask & (max_val == g)
        h[g_mask] = (60 * ((b[g_mask] - r[g_mask]) / diff[g_mask]) + 120) % 360
        
        # Blue is max
        b_mask = mask & (max_val == b)
        h[b_mask] = (60 * ((r[b_mask] - g[b_mask]) / diff[b_mask]) + 240) % 360
        
        # Saturation calculation
        s = np.where(max_val != 0, diff / max_val, 0)
        
        # Value is just the max
        v = max_val
        
        # Convert to 0-255 range like OpenCV
        h = h / 2  # OpenCV uses 0-179 for hue
        s = s * 255
        v = v * 255
        
        return np.stack([h, s, v], axis=-1).astype(np.uint8)
    
    def detect_red_object(self, rgb_img):
        """Detect red object in camera image and return its relative position"""
        if rgb_img is None:
            return 0, 0  # No steering
            
        # Convert to numpy array (remove alpha channel)
        img_array = np.array(rgb_img).reshape((self.camera_height, self.camera_width, 4))
        rgb_img = img_array[:,:,:3]  # Remove alpha channel
        
        # Convert RGB to HSV
        hsv = self.rgb_to_hsv(rgb_img)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Create mask for red color (two ranges for red in HSV)
        # Red is around 0° and 360° in hue
        red_mask1 = (h <= 10) & (s >= 50) & (v >= 50)  # Low red range
        red_mask2 = (h >= 170) & (s >= 50) & (v >= 50)  # High red range
        red_mask = red_mask1 | red_mask2
        
        # Find connected components (simple blob detection)
        if np.sum(red_mask) < 100:  # Minimum pixel threshold
            return 0, 0
        
        # Find center of mass of red pixels
        y_coords, x_coords = np.where(red_mask)
        
        if len(x_coords) == 0:
            return 0, 0
        
        # Calculate centroid
        cx = np.mean(x_coords)
        cy = np.mean(y_coords)
        
        # Convert to relative position (-1 to 1)
        rel_x = (cx - self.camera_width/2) / (self.camera_width/2)
        rel_y = (cy - self.camera_height/2) / (self.camera_height/2)
        
        return rel_x, rel_y
    
    def evaluate_locomotion(self, params, duration=5.0, use_vision=False):
        """Evaluate locomotion performance with given parameters"""
        # Reset robot position
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])
        
        # If using vision, move red cube to new random position
        if use_vision and self.red_cube_id is not None:
            new_pos = [random.uniform(-8, 8), random.uniform(-8, 8), 0.5]
            p.resetBasePositionAndOrientation(self.red_cube_id, new_pos, [0, 0, 0, 1])
        
        # Let robot stabilize
        for _ in range(100):
            p.stepSimulation()
        
        # Record initial position
        initial_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        
        # Run simulation
        steps = int(duration * 240)  # 240 Hz simulation
        
        for step in range(steps):
            time_factor = step * (1.0 / 240.0)
            
            # Get vision feedback if using vision
            steering_x, steering_y = 0, 0
            if use_vision:
                camera_img = self.get_camera_image()
                steering_x, steering_y = self.detect_red_object(camera_img)
            
            # Apply control to each joint
            for joint_idx in self.controllable_joints:
                offset, amplitude, phase = params[joint_idx]
                lower_limit = self.joint_info[joint_idx]['lower_limit']
                upper_limit = self.joint_info[joint_idx]['upper_limit']
                
                # Basic sinusoidal motion
                target = offset + amplitude * math.sin(2 * math.pi * time_factor + phase)
                
                # Apply vision-based steering if enabled
                if use_vision and steering_x != 0:
                    joint_name = self.joint_info[joint_idx]['name'].lower()
                    
                    # Modulate hip joints based on steering
                    if 'hip' in joint_name:
                        if 'hip_1' in joint_name or 'hip_4' in joint_name:  # Right side
                            target += steering_x * 0.2
                        elif 'hip_2' in joint_name or 'hip_3' in joint_name:  # Left side
                            target -= steering_x * 0.2
                
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
        
        if use_vision and self.red_cube_id is not None:
            # For vision mode, calculate distance to red cube
            cube_pos, _ = p.getBasePositionAndOrientation(self.red_cube_id)
            distance_to_target = math.sqrt(
                (final_pos[0] - cube_pos[0])**2 + 
                (final_pos[1] - cube_pos[1])**2
            )
            # Fitness is negative distance (closer is better)
            fitness = -distance_to_target
        else:
            # For locomotion mode, calculate forward speed
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
    
    def hill_climber_optimization(self, generations=50, use_vision=False):
        """Optimize locomotion parameters using hill climbing"""
        print(f"\nStarting hill climber optimization for {generations} generations...")
        print(f"Vision mode: {'ON' if use_vision else 'OFF'}")
        
        self.use_vision = use_vision
        self.learning_history = []
        
        # Initialize with random parameters
        self.initialize_random_params()
        current_params = self.locomotion_params.copy()
        current_fitness = self.evaluate_locomotion(current_params, use_vision=use_vision)
        
        self.best_speed = current_fitness
        self.best_params = current_params.copy()
        self.learning_history.append(current_fitness)
        
        print(f"Initial fitness: {current_fitness:.4f}")
        
        for generation in range(generations):
            # Create mutated version
            candidate_params = self.mutate_params(current_params)
            candidate_fitness = self.evaluate_locomotion(candidate_params, use_vision=use_vision)
            
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
        plt.ylabel('Fitness (Speed or -Distance to Target)')
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
        
        # If vision mode, reset cube position
        if self.use_vision and self.red_cube_id is not None:
            cube_pos = [random.uniform(-5, 5), random.uniform(-5, 5), 0.5]
            p.resetBasePositionAndOrientation(self.red_cube_id, cube_pos, [0, 0, 0, 1])
        
        steps = int(duration * 240)
        
        for step in range(steps):
            time_factor = step * (1.0 / 240.0)
            
            # Get vision feedback if using vision
            steering_x, steering_y = 0, 0
            if self.use_vision:
                camera_img = self.get_camera_image()
                steering_x, steering_y = self.detect_red_object(camera_img)
            
            # Apply best parameters
            for joint_idx in self.controllable_joints:
                offset, amplitude, phase = self.best_params[joint_idx]
                lower_limit = self.joint_info[joint_idx]['lower_limit']
                upper_limit = self.joint_info[joint_idx]['upper_limit']
                
                target = offset + amplitude * math.sin(2 * math.pi * time_factor + phase)
                
                # Apply vision-based steering
                if self.use_vision and steering_x != 0:
                    joint_name = self.joint_info[joint_idx]['name'].lower()
                    if 'hip' in joint_name:
                        if 'hip_1' in joint_name or 'hip_4' in joint_name:
                            target += steering_x * 0.2
                        elif 'hip_2' in joint_name or 'hip_3' in joint_name:
                            target -= steering_x * 0.2
                
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
                if self.use_vision and self.red_cube_id is not None:
                    cube_pos, _ = p.getBasePositionAndOrientation(self.red_cube_id)
                    distance = math.sqrt((robot_pos[0]-cube_pos[0])**2 + (robot_pos[1]-cube_pos[1])**2)
                    print(f"Time {step/240:.1f}s: Robot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), "
                          f"Distance to cube: {distance:.2f}")
                else:
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
    
    # Ask user for mode
    print("\nChoose mode:")
    print("1. Optimize for forward locomotion")
    print("2. Optimize for chasing red cube (with vision)")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        use_vision = choice == "2"
        
        if use_vision:
            controller.add_red_cube()
            print("Vision mode enabled - robot will learn to chase the red cube!")
        else:
            print("Locomotion mode - robot will learn to move forward fastest!")
        
        # Run optimization
        generations = 30  # Adjust as needed
        best_params = controller.hill_climber_optimization(generations=generations, 
                                                         use_vision=use_vision)
        
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