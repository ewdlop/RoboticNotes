import pybullet as p
import time
import pybullet_data
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import os

# ============================================================================
# MOTION GENERATORS
# ============================================================================

class MotionGenerator:
    """Base class for different motion generation strategies"""
    
    def __init__(self):
        self.param_names = ["param1", "param2", "param3", "param4"]
        self.description = "Base motion generator"
    
    def generate_motion(self, joint_idx, time_factor, params, joint_info):
        """Generate target position for a joint at given time"""
        raise NotImplementedError
    
    def generate_random_params(self, joint_info):
        """Generate random parameters for this motion type"""
        raise NotImplementedError
    
    def mutate_params(self, params, joint_info, mutation_rate=0.1):
        """Mutate parameters for optimization"""
        raise NotImplementedError

class SinusoidalMotion(MotionGenerator):
    def __init__(self):
        super().__init__()
        self.param_names = ["offset", "amplitude", "phase", "frequency"]
        self.description = "Classic sine wave motion"
    
    def generate_motion(self, joint_idx, time_factor, params, joint_info):
        offset, amplitude, phase, frequency = params
        return offset + amplitude * math.sin(2 * math.pi * frequency * time_factor + phase)
    
    def generate_random_params(self, joint_info):
        lower_limit = joint_info['lower_limit']
        upper_limit = joint_info['upper_limit']
        range_size = upper_limit - lower_limit
        
        offset = random.uniform(lower_limit + 0.1 * range_size, upper_limit - 0.1 * range_size)
        amplitude = random.uniform(0.1 * range_size, 0.4 * range_size)
        phase = random.uniform(0, 2 * math.pi)
        frequency = random.uniform(0.5, 2.0)
        
        return [offset, amplitude, phase, frequency]
    
    def mutate_params(self, params, joint_info, mutation_rate=0.1):
        new_params = params.copy()
        lower_limit = joint_info['lower_limit']
        upper_limit = joint_info['upper_limit']
        range_size = upper_limit - lower_limit
        
        for i in range(4):
            if random.random() < mutation_rate:
                if i == 0:  # offset
                    new_params[i] += random.gauss(0, 0.1 * range_size)
                    new_params[i] = max(lower_limit, min(upper_limit, new_params[i]))
                elif i == 1:  # amplitude
                    new_params[i] += random.gauss(0, 0.05 * range_size)
                    new_params[i] = max(0, min(0.5 * range_size, new_params[i]))
                elif i == 2:  # phase
                    new_params[i] += random.gauss(0, 0.3)
                    new_params[i] = new_params[i] % (2 * math.pi)
                elif i == 3:  # frequency
                    new_params[i] += random.gauss(0, 0.2)
                    new_params[i] = max(0.1, min(3.0, new_params[i]))
        
        return new_params

class SolitonMotion(MotionGenerator):
    def __init__(self):
        super().__init__()
        self.param_names = ["center", "amplitude", "width", "velocity"]
        self.description = "Soliton wave propagation"
    
    def generate_motion(self, joint_idx, time_factor, params, joint_info):
        center, amplitude, width, velocity = params
        # Create traveling soliton wave based on joint position
        joint_position = joint_idx / 10.0  # Normalize position
        wave_pos = velocity * time_factor - joint_position
        soliton = amplitude / (math.cosh(wave_pos / width) ** 2)
        return center + soliton
    
    def generate_random_params(self, joint_info):
        lower_limit = joint_info['lower_limit']
        upper_limit = joint_info['upper_limit']
        range_size = upper_limit - lower_limit
        
        center = random.uniform(lower_limit + 0.2 * range_size, upper_limit - 0.2 * range_size)
        amplitude = random.uniform(0.1 * range_size, 0.3 * range_size)
        width = random.uniform(0.1, 0.5)
        velocity = random.uniform(0.5, 3.0)
        
        return [center, amplitude, width, velocity]
    
    def mutate_params(self, params, joint_info, mutation_rate=0.1):
        new_params = params.copy()
        lower_limit = joint_info['lower_limit']
        upper_limit = joint_info['upper_limit']
        range_size = upper_limit - lower_limit
        
        for i in range(4):
            if random.random() < mutation_rate:
                if i == 0:  # center
                    new_params[i] += random.gauss(0, 0.1 * range_size)
                    new_params[i] = max(lower_limit, min(upper_limit, new_params[i]))
                elif i == 1:  # amplitude
                    new_params[i] += random.gauss(0, 0.05 * range_size)
                    new_params[i] = max(0, min(0.5 * range_size, new_params[i]))
                elif i == 2:  # width
                    new_params[i] += random.gauss(0, 0.1)
                    new_params[i] = max(0.05, min(1.0, new_params[i]))
                elif i == 3:  # velocity
                    new_params[i] += random.gauss(0, 0.3)
                    new_params[i] = max(0.1, min(5.0, new_params[i]))
        
        return new_params

class CPGMotion(MotionGenerator):
    def __init__(self):
        super().__init__()
        self.param_names = ["offset", "amplitude", "coupling", "phase_shift"]
        self.description = "Central Pattern Generator"
        self.joint_states = {}
    
    def generate_motion(self, joint_idx, time_factor, params, joint_info):
        offset, amplitude, coupling, phase_shift = params
        
        # Basic oscillation
        base_motion = amplitude * math.sin(2 * math.pi * time_factor)
        
        # Coupling with neighboring joints (simplified)
        neighbor_influence = 0
        neighbor_phase = phase_shift
        neighbor_influence = coupling * math.sin(2 * math.pi * time_factor + neighbor_phase)
        
        return offset + base_motion + neighbor_influence
    
    def generate_random_params(self, joint_info):
        lower_limit = joint_info['lower_limit']
        upper_limit = joint_info['upper_limit']
        range_size = upper_limit - lower_limit
        
        offset = random.uniform(lower_limit + 0.1 * range_size, upper_limit - 0.1 * range_size)
        amplitude = random.uniform(0.1 * range_size, 0.3 * range_size)
        coupling = random.uniform(0.05, 0.2)
        phase_shift = random.uniform(0, 2 * math.pi)
        
        return [offset, amplitude, coupling, phase_shift]
    
    def mutate_params(self, params, joint_info, mutation_rate=0.1):
        new_params = params.copy()
        lower_limit = joint_info['lower_limit']
        upper_limit = joint_info['upper_limit']
        range_size = upper_limit - lower_limit
        
        for i in range(4):
            if random.random() < mutation_rate:
                if i == 0:  # offset
                    new_params[i] += random.gauss(0, 0.1 * range_size)
                    new_params[i] = max(lower_limit, min(upper_limit, new_params[i]))
                elif i == 1:  # amplitude
                    new_params[i] += random.gauss(0, 0.05 * range_size)
                    new_params[i] = max(0, min(0.5 * range_size, new_params[i]))
                elif i == 2:  # coupling
                    new_params[i] += random.gauss(0, 0.05)
                    new_params[i] = max(0, min(0.5, new_params[i]))
                elif i == 3:  # phase_shift
                    new_params[i] += random.gauss(0, 0.3)
                    new_params[i] = new_params[i] % (2 * math.pi)
        
        return new_params

class PulseMotion(MotionGenerator):
    def __init__(self):
        super().__init__()
        self.param_names = ["baseline", "peak_amp", "pulse_width", "frequency"]
        self.description = "Pulse wave motion"
    
    def generate_motion(self, joint_idx, time_factor, params, joint_info):
        baseline, peak_amplitude, pulse_width, frequency = params
        pulse_phase = (frequency * time_factor) % 1.0
        
        if pulse_phase < pulse_width:
            pulse_strength = math.sin(math.pi * pulse_phase / pulse_width)
            return baseline + peak_amplitude * pulse_strength
        else:
            return baseline
    
    def generate_random_params(self, joint_info):
        lower_limit = joint_info['lower_limit']
        upper_limit = joint_info['upper_limit']
        range_size = upper_limit - lower_limit
        
        baseline = random.uniform(lower_limit + 0.1 * range_size, upper_limit - 0.1 * range_size)
        peak_amplitude = random.uniform(0.1 * range_size, 0.4 * range_size)
        pulse_width = random.uniform(0.1, 0.5)
        frequency = random.uniform(0.5, 2.0)
        
        return [baseline, peak_amplitude, pulse_width, frequency]
    
    def mutate_params(self, params, joint_info, mutation_rate=0.1):
        new_params = params.copy()
        lower_limit = joint_info['lower_limit']
        upper_limit = joint_info['upper_limit']
        range_size = upper_limit - lower_limit
        
        for i in range(4):
            if random.random() < mutation_rate:
                if i == 0:  # baseline
                    new_params[i] += random.gauss(0, 0.1 * range_size)
                    new_params[i] = max(lower_limit, min(upper_limit, new_params[i]))
                elif i == 1:  # peak_amplitude
                    new_params[i] += random.gauss(0, 0.05 * range_size)
                    new_params[i] = max(0, min(0.5 * range_size, new_params[i]))
                else:  # pulse_width or frequency
                    new_params[i] += random.gauss(0, 0.1)
                    new_params[i] = max(0.1, min(3.0, new_params[i]))
        
        return new_params

class ChirpMotion(MotionGenerator):
    def __init__(self):
        super().__init__()
        self.param_names = ["offset", "amplitude", "start_freq", "freq_sweep"]
        self.description = "Frequency modulated chirp"
    
    def generate_motion(self, joint_idx, time_factor, params, joint_info):
        offset, amplitude, start_freq, freq_sweep = params
        instantaneous_freq = start_freq + freq_sweep * time_factor
        phase = 2 * math.pi * (start_freq * time_factor + 0.5 * freq_sweep * time_factor**2)
        return offset + amplitude * math.sin(phase)
    
    def generate_random_params(self, joint_info):
        lower_limit = joint_info['lower_limit']
        upper_limit = joint_info['upper_limit']
        range_size = upper_limit - lower_limit
        
        offset = random.uniform(lower_limit + 0.1 * range_size, upper_limit - 0.1 * range_size)
        amplitude = random.uniform(0.1 * range_size, 0.3 * range_size)
        start_freq = random.uniform(0.2, 1.0)
        freq_sweep = random.uniform(-0.5, 0.5)
        
        return [offset, amplitude, start_freq, freq_sweep]
    
    def mutate_params(self, params, joint_info, mutation_rate=0.1):
        new_params = params.copy()
        lower_limit = joint_info['lower_limit']
        upper_limit = joint_info['upper_limit']
        range_size = upper_limit - lower_limit
        
        for i in range(4):
            if random.random() < mutation_rate:
                if i == 0:  # offset
                    new_params[i] += random.gauss(0, 0.1 * range_size)
                    new_params[i] = max(lower_limit, min(upper_limit, new_params[i]))
                elif i == 1:  # amplitude
                    new_params[i] += random.gauss(0, 0.05 * range_size)
                    new_params[i] = max(0, min(0.5 * range_size, new_params[i]))
                else:  # frequencies
                    new_params[i] += random.gauss(0, 0.2)
                    new_params[i] = max(-2.0, min(2.0, new_params[i]))
        
        return new_params

class ChaoticMotion(MotionGenerator):
    def __init__(self):
        super().__init__()
        self.param_names = ["offset", "amplitude", "param_a", "param_b"]
        self.description = "Chaotic attractor motion"
        self.chaotic_states = {}
    
    def generate_motion(self, joint_idx, time_factor, params, joint_info):
        offset, amplitude, param_a, param_b = params
        
        if joint_idx not in self.chaotic_states:
            self.chaotic_states[joint_idx] = 0.5
        
        # Update chaotic state using logistic map with external forcing
        x = self.chaotic_states[joint_idx]
        self.chaotic_states[joint_idx] = param_a * x * (1 - x) + param_b * math.sin(2 * math.pi * time_factor)
        self.chaotic_states[joint_idx] = max(0, min(1, self.chaotic_states[joint_idx]))
        
        return offset + amplitude * (2 * self.chaotic_states[joint_idx] - 1)
    
    def generate_random_params(self, joint_info):
        lower_limit = joint_info['lower_limit']
        upper_limit = joint_info['upper_limit']
        range_size = upper_limit - lower_limit
        
        offset = random.uniform(lower_limit + 0.1 * range_size, upper_limit - 0.1 * range_size)
        amplitude = random.uniform(0.1 * range_size, 0.3 * range_size)
        param_a = random.uniform(2.5, 4.0)  # Chaotic regime
        param_b = random.uniform(0.1, 0.3)
        
        return [offset, amplitude, param_a, param_b]
    
    def mutate_params(self, params, joint_info, mutation_rate=0.1):
        new_params = params.copy()
        lower_limit = joint_info['lower_limit']
        upper_limit = joint_info['upper_limit']
        range_size = upper_limit - lower_limit
        
        for i in range(4):
            if random.random() < mutation_rate:
                if i == 0:  # offset
                    new_params[i] += random.gauss(0, 0.1 * range_size)
                    new_params[i] = max(lower_limit, min(upper_limit, new_params[i]))
                elif i == 1:  # amplitude
                    new_params[i] += random.gauss(0, 0.05 * range_size)
                    new_params[i] = max(0, min(0.5 * range_size, new_params[i]))
                elif i == 2:  # param_a
                    new_params[i] += random.gauss(0, 0.2)
                    new_params[i] = max(1.0, min(4.0, new_params[i]))
                elif i == 3:  # param_b
                    new_params[i] += random.gauss(0, 0.1)
                    new_params[i] = max(0, min(1.0, new_params[i]))
        
        return new_params

# ============================================================================
# MOTION GENERATOR FACTORY
# ============================================================================

class MotionFactory:
    """Factory to create motion generators"""
    
    @staticmethod
    def get_available_motions():
        return {
            "sinusoidal": SinusoidalMotion(),
            "soliton": SolitonMotion(),
            "cpg": CPGMotion(),
            "pulse": PulseMotion(),
            "chirp": ChirpMotion(),
            "chaotic": ChaoticMotion()
        }
    
    @staticmethod
    def create_motion_generator(motion_type):
        motions = MotionFactory.get_available_motions()
        return motions.get(motion_type, SinusoidalMotion())

# ============================================================================
# PARAMETER STORAGE MANAGER
# ============================================================================

class ParameterManager:
    """Handles saving and loading of optimization parameters"""
    
    def __init__(self, filename="best_robot_params.json"):
        self.filename = filename
    
    def save_parameters(self, best_speed, motion_type, best_params, joint_info, learning_history):
        """Save optimization results to file"""
        try:
            params_data = {
                'best_speed': best_speed,
                'motion_type': motion_type,
                'best_params': {str(k): v for k, v in best_params.items()},
                'joint_info': {str(k): v for k, v in joint_info.items()},
                'learning_history': learning_history
            }
            
            with open(self.filename, 'w') as f:
                json.dump(params_data, f, indent=2)
            
            print(f"Parameters saved to {self.filename}")
            print(f"Best speed: {best_speed:.4f} m/s")
            return True
            
        except Exception as e:
            print(f"Error saving parameters: {e}")
            return False
    
    def load_parameters(self):
        """Load optimization results from file"""
        if not os.path.exists(self.filename):
            return None
            
        try:
            with open(self.filename, 'r') as f:
                params_data = json.load(f)
            
            # Convert string keys back to integers
            params_data['best_params'] = {int(k): v for k, v in params_data['best_params'].items()}
            
            print(f"Parameters loaded from {self.filename}")
            print(f"Motion type: {params_data.get('motion_type', 'unknown')}")
            print(f"Best speed: {params_data['best_speed']:.4f} m/s")
            
            return params_data
            
        except Exception as e:
            print(f"Error loading parameters: {e}")
            return None
    
    def print_info(self):
        """Print information about saved parameters"""
        if not os.path.exists(self.filename):
            print(f"No saved parameters found at {self.filename}")
            return
            
        try:
            with open(self.filename, 'r') as f:
                params_data = json.load(f)
            
            motion_type = params_data.get('motion_type', 'unknown')
            motion_gen = MotionFactory.create_motion_generator(motion_type)
            
            print(f"\nSaved Parameters Info:")
            print(f"  File: {self.filename}")
            print(f"  Motion Type: {motion_type} - {motion_gen.description}")
            print(f"  Best Speed: {params_data['best_speed']:.4f} m/s")
            print(f"  Number of joints: {len(params_data['best_params'])}")
            print(f"  Learning history length: {len(params_data.get('learning_history', []))}")
            
            print("  Joint parameters:")
            joint_info = params_data.get('joint_info', {})
            for joint_idx_str, params in params_data['best_params'].items():
                joint_name = joint_info.get(joint_idx_str, {}).get('name', f'Joint_{joint_idx_str}')
                param_str = ', '.join([f'{name}={val:.3f}' for name, val in zip(motion_gen.param_names, params)])
                print(f"    {joint_name}: {param_str}")
                
        except Exception as e:
            print(f"Error reading saved parameters: {e}")

# ============================================================================
# ROBOT CONTROLLER
# ============================================================================

class RobotController:
    """Main robot control and optimization system"""
    
    def __init__(self):
        # Simulation objects
        self.physicsClient = None
        self.robotId = None
        self.planeId = None
        
        # 目標立方體屬性
        self.target_position = None
        self.cubeId = None
        
        # Robot analysis
        self.controllable_joints = []
        self.joint_info = {}
        
        # Motion generation
        self.motion_generator = None
        self.locomotion_params = {}
        
        # Optimization tracking
        self.learning_history = []
        self.best_speed = -float('inf')
        self.best_params = None
        
        # Managers
        self.param_manager = ParameterManager()
    
    def setup_simulation(self):
        """Initialize PyBullet simulation environment"""
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")
        
        # --- 新增紅色立方體目標 ---
        cube_half_extents = [0.1, 0.1, 0.1]
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=cube_half_extents,
            rgbaColor=[1, 0, 0, 1]
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=cube_half_extents
        )
        # 將目標位置儲存
        self.target_position = [3, 0, 0.5]
        # 質量 0 → 靜止
        self.cubeId = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.target_position
        )
        # --- 結束新增 ---
        
        print("Simulation environment initialized")
    
    def load_robot(self):
        """Load the ant robot from MJCF file"""
        try:
            robotIds = p.loadMJCF("ant.xml")
            print(f"MJCF loaded successfully. Number of bodies: {len(robotIds)}")
            
            if len(robotIds) > 1:
                self.robotId = robotIds[1]
            else:
                self.robotId = robotIds[0]
                
            # Set initial position
            p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])
            
            basePos, baseOrn = p.getBasePositionAndOrientation(self.robotId)
            print(f"Robot loaded at position: {basePos}")
            return True
            
        except Exception as e:
            print(f"Failed to load robot: {e}")
            return False
    
    def analyze_joints(self):
        """Analyze and categorize all joints in the robot"""
        numJoints = p.getNumJoints(self.robotId)
        print(f"\nAnalyzing {numJoints} joints...")
        
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
            
            # Include hinge joints but not the root free joint
            if jointType == 0 and jointName != 'root':
                if jointLowerLimit < jointUpperLimit:
                    self.controllable_joints.append(i)
                    print(f"  Controllable joint {i}: {jointName}")
        
        print(f"Found {len(self.controllable_joints)} controllable joints")
        return len(self.controllable_joints) > 0
    
    def set_motion_type(self, motion_type):
        """Set the motion generation type"""
        self.motion_generator = MotionFactory.create_motion_generator(motion_type)
        print(f"Motion type set to: {motion_type} - {self.motion_generator.description}")
    
    def initialize_random_parameters(self):
        """Initialize random parameters for all joints"""
        if not self.motion_generator:
            raise ValueError("Motion generator not set. Call set_motion_type() first.")
        
        self.locomotion_params = {}
        print(f"\nInitializing random parameters for {self.motion_generator.description}:")
        
        for joint_idx in self.controllable_joints:
            joint_name = self.joint_info[joint_idx]['name']
            params = self.motion_generator.generate_random_params(self.joint_info[joint_idx])
            self.locomotion_params[joint_idx] = params
            
            param_str = ', '.join([f'{name}={val:.3f}' for name, val in zip(self.motion_generator.param_names, params)])
            print(f"  {joint_name}: {param_str}")
    
    def evaluate_fitness(self, params, duration=5.0):
        """Evaluate locomotion performance by how fast the robot approaches the cube"""
        # Reset robot位置和姿態
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])
        
        # Stabilization period
        for _ in range(100):
            p.stepSimulation()
        
        # 1. 初始時，計算 robot 與紅色立方體的距離
        robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        dx = robot_pos[0] - self.target_position[0]
        dy = robot_pos[1] - self.target_position[1]
        dz = robot_pos[2] - self.target_position[2]
        initial_dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Run simulation
        steps = int(duration * 240)  # 240 Hz
        for step in range(steps):
            time_factor = step * (1.0 / 240.0)
            
            # Apply motion to each joint
            for joint_idx in self.controllable_joints:
                joint_params = params[joint_idx]
                target = self.motion_generator.generate_motion(
                    joint_idx, time_factor, joint_params, self.joint_info[joint_idx]
                )
                # Clamp to joint limits
                lower_limit = self.joint_info[joint_idx]['lower_limit']
                upper_limit = self.joint_info[joint_idx]['upper_limit']
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
        
        # 2. 最終時，計算 robot 與紅色立方體的距離
        final_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        dx2 = final_pos[0] - self.target_position[0]
        dy2 = final_pos[1] - self.target_position[1]
        dz2 = final_pos[2] - self.target_position[2]
        final_dist = math.sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2)
        
        # 3. 健康度：平均每秒縮短的距離
        fitness = (initial_dist - final_dist) / duration
        return fitness
    
    def optimize_parameters(self, generations=50):
        """Optimize parameters using hill climbing algorithm"""
        print(f"\nStarting optimization for {generations} generations...")
        
        self.learning_history = []
        
        # Initialize with random parameters
        self.initialize_random_parameters()
        current_params = self.locomotion_params.copy()
        current_fitness = self.evaluate_fitness(current_params)
        
        self.best_speed = current_fitness
        self.best_params = current_params.copy()
        self.learning_history.append(current_fitness)
        
        print(f"Initial fitness: {current_fitness:.4f} m/s")
        
        # Hill climbing optimization
        for generation in range(generations):
            # Create mutated candidate
            candidate_params = self.mutate_parameters(current_params)
            candidate_fitness = self.evaluate_fitness(candidate_params)
            
            # Accept if better
            if candidate_fitness > current_fitness:
                current_params = candidate_params
                current_fitness = candidate_fitness
                print(f"Generation {generation+1}: NEW BEST fitness = {current_fitness:.4f} m/s")
                
                if current_fitness > self.best_speed:
                    self.best_speed = current_fitness
                    self.best_params = current_params.copy()
            else:
                print(f"Generation {generation+1}: fitness = {candidate_fitness:.4f} m/s (no improvement)")
            
            self.learning_history.append(current_fitness)
        
        print(f"\nOptimization complete! Best fitness: {self.best_speed:.4f} m/s")
        
        # Save results
        motion_type = type(self.motion_generator).__name__.replace('Motion', '').lower()
        self.param_manager.save_parameters(
            self.best_speed, motion_type, self.best_params, 
            self.joint_info, self.learning_history
        )
        
        return self.best_params
    
    def mutate_parameters(self, params, mutation_rate=0.1):
        """Create mutated version of parameters"""
        new_params = {}
        
        for joint_idx in params:
            joint_info = self.joint_info[joint_idx]
            new_params[joint_idx] = self.motion_generator.mutate_params(
                params[joint_idx], joint_info, mutation_rate
            )
        
        return new_params
    
    def load_saved_parameters(self):
        """Load previously saved parameters"""
        params_data = self.param_manager.load_parameters()
        if params_data is None:
            return False
        
        # Set motion type and parameters
        motion_type = params_data.get('motion_type', 'sinusoidal')
        self.set_motion_type(motion_type)
        
        self.best_speed = params_data['best_speed']
        self.best_params = params_data['best_params']
        
        # 驗證關節是否一致
        current_joints = set(self.controllable_joints)
        loaded_joints = set(self.best_params.keys())
        if loaded_joints != current_joints:
            print("Warning: Loaded parameters don't match current robot joints!")
            return False
        
        return True
    
    def demonstrate_robot(self, duration=15.0):
        """Demonstrate the robot using optimized parameters"""
        if self.best_params is None:
            print("No optimized parameters available!")
            return
        
        print(f"\nDemonstrating optimized robot for {duration} seconds...")
        motion_type = type(self.motion_generator).__name__.replace('Motion', '').lower()
        print(f"Motion type: {motion_type}")
        
        print("Optimized parameters:")
        for joint_idx, params in self.best_params.items():
            joint_name = self.joint_info[joint_idx]['name']
            param_str = ', '.join([f'{name}={val:.3f}' for name, val in zip(self.motion_generator.param_names, params)])
            print(f"  {joint_name}: {param_str}")
        
        # Reset robot
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])
        
        steps = int(duration * 240)
        
        for step in range(steps):
            time_factor = step * (1.0 / 240.0)
            
            # Apply optimized motion
            for joint_idx in self.controllable_joints:
                joint_params = self.best_params[joint_idx]
                target = self.motion_generator.generate_motion(
                    joint_idx, time_factor, joint_params, self.joint_info[joint_idx]
                )
                
                # Clamp to joint limits
                lower_limit = self.joint_info[joint_idx]['lower_limit']
                upper_limit = self.joint_info[joint_idx]['upper_limit']
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
            
            # Status updates
            if step % 480 == 0:
                robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)
                print(f"Time {step/240:.1f}s: Robot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})")
    
    def plot_learning_curve(self):
        """Plot the optimization learning curve"""
        if not self.learning_history:
            print("No learning history to plot!")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.learning_history, linewidth=2, color='blue')
        plt.title('Robot Locomotion Optimization Learning Curve', fontsize=14)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness (Speed m/s)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        initial_fitness = self.learning_history[0]
        final_fitness = self.learning_history[-1]
        best_fitness = max(self.learning_history)
        improvement = final_fitness - initial_fitness
        
        plt.text(0.02, 0.98, f'Initial: {initial_fitness:.3f} m/s\n'
                            f'Final: {final_fitness:.3f} m/s\n'
                            f'Best: {best_fitness:.3f} m/s\n'
                            f'Improvement: {improvement:.3f} m/s',
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nLearning Statistics:")
        print(f"  Initial fitness: {initial_fitness:.4f} m/s")
        print(f"  Final fitness: {final_fitness:.4f} m/s")
        print(f"  Best fitness: {best_fitness:.4f} m/s")
        print(f"  Total improvement: {improvement:.4f} m/s")
        print(f"  Improvement percentage: {(improvement/initial_fitness)*100:.1f}%")
    
    def continue_optimization(self, additional_generations=20):
        """Continue optimization from current best parameters"""
        if self.best_params is None:
            print("No existing parameters to continue from!")
            return False
        
        print(f"Continuing optimization for {additional_generations} more generations...")
        print(f"Starting from fitness: {self.best_speed:.4f} m/s")
        
        # Continue from best parameters
        current_params = self.best_params.copy()
        current_fitness = self.best_speed
        
        for generation in range(additional_generations):
            # Create mutated candidate
            candidate_params = self.mutate_parameters(current_params)
            candidate_fitness = self.evaluate_fitness(candidate_params)
            
            # Accept if better
            if candidate_fitness > current_fitness:
                current_params = candidate_params
                current_fitness = candidate_fitness
                print(f"Generation {len(self.learning_history)+1}: NEW BEST fitness = {current_fitness:.4f} m/s")
                
                if current_fitness > self.best_speed:
                    self.best_speed = current_fitness
                    self.best_params = current_params.copy()
            else:
                print(f"Generation {len(self.learning_history)+1}: fitness = {candidate_fitness:.4f} m/s (no improvement)")
            
            self.learning_history.append(current_fitness)
        
        print(f"Continued optimization complete! Best fitness: {self.best_speed:.4f} m/s")
        
        # Save updated results
        motion_type = type(self.motion_generator).__name__.replace('Motion', '').lower()
        self.param_manager.save_parameters(
            self.best_speed, motion_type, self.best_params, 
            self.joint_info, self.learning_history
        )
        
        return True

# ============================================================================
# USER INTERFACE
# ============================================================================

class UserInterface:
    """Handles user interaction and program flow"""
    
    @staticmethod
    def select_motion_type():
        """Allow user to select motion type"""
        motions = MotionFactory.get_available_motions()
        
        print(f"\nAvailable motion types:")
        motion_list = list(motions.items())
        for i, (motion_type, motion_gen) in enumerate(motion_list, 1):
            print(f"{i}. {motion_type}: {motion_gen.description}")
        
        try:
            choice = input(f"\nSelect motion type (1-{len(motion_list)}, default=1): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(motion_list):
                selected_type = motion_list[int(choice) - 1][0]
            else:
                selected_type = "sinusoidal"  # Default
                
            print(f"Selected: {selected_type}")
            return selected_type
            
        except:
            print("Using default: sinusoidal")
            return "sinusoidal"
    
    @staticmethod
    def get_user_choice():
        """Get user's choice for what to do"""
        print("\nChoose an option:")
        print("1. Run new optimization")
        print("2. Load saved parameters and demonstrate")
        print("3. Load saved parameters and continue optimization")
        
        try:
            choice = input("Enter choice (1, 2, or 3): ").strip()
            return choice
        except:
            return "1"
    
    @staticmethod
    def get_generations(default=30):
        """Get number of generations from user"""
        try:
            generations = input(f"Enter number of generations (default {default}): ").strip()
            return int(generations) if generations.isdigit() else default
        except:
            return default

# ============================================================================
# MAIN APPLICATION - ENTRY POINT
# ============================================================================

def main():
    """Main application entry point - orchestrates the entire program flow"""
    print("=" * 70)
    print("ROBOT LOCOMOTION OPTIMIZATION SYSTEM")
    print("=" * 70)
    
    # Initialize the main controller
    controller = RobotController()
    
    try:
        # STEP 1: Setup simulation environment
        print("\n1. Setting up simulation environment...")
        controller.setup_simulation()
        
        # STEP 2: Load robot from MJCF file
        print("2. Loading robot...")
        if not controller.load_robot():
            print("ERROR: Failed to load robot!")
            return
        
        # STEP 3: Analyze robot joints and find controllable ones
        print("3. Analyzing robot joints...")
        if not controller.analyze_joints():
            print("ERROR: No controllable joints found!")
            return
        
        # STEP 4: Check for any previously saved optimization results
        print("4. Checking for saved parameters...")
        controller.param_manager.print_info()
        
        # STEP 5: Let user select which motion type to use
        print("5. Selecting motion type...")
        motion_type = UserInterface.select_motion_type()
        controller.set_motion_type(motion_type)
        
        # STEP 6: Get user's choice of what to do
        choice = UserInterface.get_user_choice()
        
        # OPTION 2: Load saved parameters and demonstrate
        if choice == "2":
            print("\n6. Loading saved parameters...")
            if controller.load_saved_parameters():
                print("7. Demonstrating optimized robot...")
                controller.demonstrate_robot(duration=15.0)
            else:
                print("Failed to load parameters. Running new optimization instead.")
                choice = "1"  # Fall back to new optimization
        
        # OPTION 3: Load saved parameters and continue optimizing
        elif choice == "3":
            print("\n6. Loading saved parameters...")
            if controller.load_saved_parameters():
                print("7. Continuing optimization...")
                additional_gens = UserInterface.get_generations(20)
                controller.continue_optimization(additional_gens)
                
                print("8. Plotting learning curve...")
                controller.plot_learning_curve()
                
                print("\nPress Enter to see demonstration...")
                input()
                print("9. Demonstrating optimized robot...")
                controller.demonstrate_robot(duration=15.0)
            else:
                print("Failed to load parameters. Running new optimization instead.")
                choice = "1"  # Fall back to new optimization
        
        # OPTION 1: Run completely new optimization
        if choice == "1":
            print(f"\n6. Starting new optimization with {motion_type} motion...")
            generations = UserInterface.get_generations(30)
            
            print("7. Optimizing parameters...")
            controller.optimize_parameters(generations=generations)
            
            print("8. Plotting learning curve...")
            controller.plot_learning_curve()
            
            print("\nPress Enter to see demonstration...")
            input()
            print("9. Demonstrating optimized robot...")
            controller.demonstrate_robot(duration=15.0)
        
        # Program completion
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE!")
        print(f"Best speed achieved: {controller.best_speed:.4f} m/s")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\nERROR occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always clean up the simulation
        print("\nCleaning up simulation...")
        if controller.physicsClient is not None:
            p.disconnect()
        print("Goodbye!")

# Program entry point
if __name__ == "__main__":
    main()
