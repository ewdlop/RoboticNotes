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
        cube_size = 0.2
        cube_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size/2]*3)
        cube_visual_shape = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[cube_size/2]*3, 
            rgbaColor=[1, 0, 0, 1]  # Red color
        )
        
        self.cubeId = p.createMultiBody(
            baseMass=0,  # Mass = 0 makes it static
            baseCollisionShapeIndex=cube_collision_shape,
            baseVisualShapeIndex=cube_visual_shape,
            basePosition=self.cube_position
        )
        
        # Make cube completely immovable by fixing it in place
        p.createConstraint(
            self.cubeId, -1, -1, -1, 
            p.JOINT_FIXED, [0, 0, 0], 
            [0, 0, 0], 
            self.cube_position
        )
        
        print(f"Red target cube created (immovable) at position: {self.cube_position}")
        
    def update_cube_position(self, simulation_time):
        """Update cube position based on movement pattern (若之後需要動態，可自行再開啟)"""
        # 目前沒有啟用移動邏輯
        pass
        
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
            
            # Set initial position（高度 1m，稍微高一些避免一放就跪地）
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
            
            # Hinge joints且非 root
            if jointType == p.JOINT_REVOLUTE and jointLowerLimit < jointUpperLimit:
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
            
            range_size = upper_limit - lower_limit
            # offset：放在中間附近
            offset = (lower_limit + upper_limit) / 2.0  
            # 振幅改為 [0.3*range, 0.9*range]
            amplitude = random.uniform(0.3 * range_size, 0.9 * range_size)
            phase = random.uniform(0, 2 * math.pi)
            frequency = random.uniform(0.5, 2.0)  # Hz
            
            self.locomotion_params[joint_idx] = [offset, amplitude, phase, frequency]
            print(f"Joint {joint_idx} ({joint_name}): offset={offset:.3f}, amp={amplitude:.3f}, phase={phase:.3f}, freq={frequency:.3f}")

    def calculate_fitness(self, robot_positions, cube_positions, duration):
        """不再使用此函數，改用 evaluate_locomotion 回傳值即可"""
        pass

    def evaluate_locomotion(self, params, duration=8.0):
        """
        Evaluate locomotion performance with cube following
        1. 先 reset 機器人到平衡站立姿態
        2. 再從第 N 步開始依據 params 做 sin() 運動
        3. 最後計算 avg_speed 與 final_distance，回傳 raw fitness（可為負值）
        """
        # 1. 先把機器人 reset 到站立
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])
        # 先把所有關節設成「站立中點」
        for joint_idx in self.controllable_joints:
            lower = self.joint_info[joint_idx]['lower_limit']
            upper = self.joint_info[joint_idx]['upper_limit']
            mid = (lower + upper) / 2.0
            p.resetJointState(self.robotId, joint_idx, mid)
        # 執行幾步讓機器人穩定
        for _ in range(100):
            p.stepSimulation()
        
        # 2. 開始正式模擬
        robot_positions = []
        steps = int(duration * 240)  # 240 Hz
        
        for step in range(steps):
            t = step / 240.0
            
            # 取得機器人底盤位置
            robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)
            cube_pos = self.cube_position
            
            # 記錄每 10 步的位置
            if step % 10 == 0:
                robot_positions.append(robot_pos)
            
            # 計算朝向紅色立方體的方向
            dx = cube_pos[0] - robot_pos[0]
            dy = cube_pos[1] - robot_pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0:
                dir_x = dx / dist
                dir_y = dy / dist
            else:
                dir_x = dir_y = 0
            
            # 針對每個可控制關節下 control
            for joint_idx in self.controllable_joints:
                offset, amplitude, phase, freq = params[joint_idx]
                lower = self.joint_info[joint_idx]['lower_limit']
                upper = self.joint_info[joint_idx]['upper_limit']
                
                # sin() base motion
                base_motion = offset + amplitude * math.sin(2*math.pi*freq*t + phase)
                
                # 加上方向偏移(hip 用)
                joint_name = self.joint_info[joint_idx]['name'].lower()
                bias = 0
                if 'hip' in joint_name and dist > 0.5:
                    # 前腿往 x 正向偏移，後腿往 x 負向偏移
                    if any(k in joint_name for k in ['front','0','1']):
                        bias = 0.2 * dir_x * (dist / 5.0)
                    else:
                        bias = -0.2 * dir_x * (dist / 5.0)
                
                target = base_motion + bias
                # 限制在關節範圍
                target = max(lower, min(upper, target))
                
                p.setJointMotorControl2(
                    bodyUniqueId=self.robotId,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=200,
                    maxVelocity=3.0
                )
            
            p.stepSimulation()
            # 取消 time.sleep 可加速 batch 評估
            # time.sleep(1.0/240.0)
        
        # 3. 計算 fitness：用速度和最終距離
        # 最終位置
        final_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        final_dist = math.sqrt(
            (final_pos[0] - cube_pos[0])**2 + 
            (final_pos[1] - cube_pos[1])**2
        )
        
        # 計算平均速度（每 10 步記錄一次）
        if len(robot_positions) >= 2:
            total_d = 0
            for i in range(1, len(robot_positions)):
                dx2 = robot_positions[i][0] - robot_positions[i-1][0]
                dy2 = robot_positions[i][1] - robot_positions[i-1][1]
                total_d += math.sqrt(dx2*dx2 + dy2*dy2)
            avg_speed = total_d / duration
        else:
            avg_speed = 0.0
        
        # 新的 fitness：速度 - 0.5 * 距離
        fitness = avg_speed - 0.5 * final_dist
        
        return fitness  # **保留負值，不做截斷**
    
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
                        new_params[joint_idx][i] += random.gauss(0, 0.05 * range_size)
                        # 將 offset 保持在關節範圍
                        new_params[joint_idx][i] = max(lower_limit, 
                                                      min(upper_limit, new_params[joint_idx][i]))
                    elif i == 1:  # amplitude
                        new_params[joint_idx][i] += random.gauss(0, 0.1 * range_size)
                        # 振幅限制為 [0, range_size]
                        new_params[joint_idx][i] = max(0, 
                                                      min(range_size, new_params[joint_idx][i]))
                    elif i == 2:  # phase
                        new_params[joint_idx][i] += random.gauss(0, 0.3)
                        new_params[joint_idx][i] %= (2 * math.pi)
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
            
            self.best_fitness = params_data['best_fitness']
            self.best_params = {int(k): v for k, v in params_data['best_params'].items()}
            self.learning_history = params_data.get('learning_history', [])
            self.cube_position = params_data.get('cube_position', [3, 0, 0.5])
            
            print(f"Best parameters loaded from {self.params_file}")
            print(f"Loaded best fitness: {self.best_fitness:.4f}")
            
            # 確認 loaded_joints 與當前可控制關節一致
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
        
        for gen in range(generations):
            candidate_params = self.mutate_params(current_params)
            candidate_fitness = self.evaluate_locomotion(candidate_params)
            
            if candidate_fitness > current_fitness:
                current_params = candidate_params
                current_fitness = candidate_fitness
                print(f"Generation {gen+1}: NEW BEST fitness = {current_fitness:.4f}")
                
                if current_fitness > self.best_fitness:
                    self.best_fitness = current_fitness
                    self.best_params = current_params.copy()
            else:
                print(f"Generation {gen+1}: fitness = {candidate_fitness:.4f} (no improvement)")
            
            self.learning_history.append(current_fitness)
        
        print(f"\nOptimization complete!")
        print(f"Best fitness achieved: {self.best_fitness:.4f}")
        
        # 自動存檔
        self.save_best_params()
        
        return self.best_params
    
    def plot_learning_curve(self):
        """Plot the learning curve"""
        if not self.learning_history:
            print("No learning history to plot!")
            return
            
        plt.figure(figsize=(12, 8))
        
        # 主 learning curve
        plt.subplot(2, 1, 1)
        plt.plot(self.learning_history, linewidth=2)
        plt.title('Robot Cube-Following Learning Curve')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.grid(True, alpha=0.3)
        
        # 若有足夠資料，繪製移動平均
        if len(self.learning_history) > 10:
            window = min(10, len(self.learning_history) // 4)
            moving_avg = np.convolve(self.learning_history, np.ones(window)/window, mode='valid')
            plt.plot(
                range(window-1, len(self.learning_history)), 
                moving_avg, 
                linewidth=2, alpha=0.7, label=f'Moving Avg ({window})'
            )
            plt.legend()
        
        # 改善幅度 histogram
        plt.subplot(2, 1, 2)
        improvements = [
            self.learning_history[i] - self.learning_history[i-1] 
            for i in range(1, len(self.learning_history))
        ]
        plt.hist(improvements, bins=20, alpha=0.7)
        plt.title('Distribution of Fitness Improvements')
        plt.xlabel('Fitness Change')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 印出統計數據
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
        
        # Reset robot 到站立姿態
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])
        for joint_idx in self.controllable_joints:
            lower = self.joint_info[joint_idx]['lower_limit']
            upper = self.joint_info[joint_idx]['upper_limit']
            mid = (lower + upper) / 2.0
            p.resetJointState(self.robotId, joint_idx, mid)
        for _ in range(100):
            p.stepSimulation()
        
        steps = int(duration * 240)
        for step in range(steps):
            t = step / 240.0
            
            robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)
            cube_pos = self.cube_position
            
            dx = cube_pos[0] - robot_pos[0]
            dy = cube_pos[1] - robot_pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0:
                dir_x = dx / dist
                dir_y = dy / dist
            else:
                dir_x = dir_y = 0
            
            for joint_idx in self.controllable_joints:
                offset, amplitude, phase, freq = self.best_params[joint_idx]
                lower = self.joint_info[joint_idx]['lower_limit']
                upper = self.joint_info[joint_idx]['upper_limit']
                
                base_motion = offset + amplitude * math.sin(2*math.pi*freq*t + phase)
                
                bias = 0
                joint_name = self.joint_info[joint_idx]['name'].lower()
                if 'hip' in joint_name and dist > 0.5:
                    if any(k in joint_name for k in ['front','0','1']):
                        bias = 0.2 * dir_x * (dist / 5.0)
                    else:
                        bias = -0.2 * dir_x * (dist / 5.0)
                
                target = base_motion + bias
                target = max(lower, min(upper, target))
                
                p.setJointMotorControl2(
                    bodyUniqueId=self.robotId,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target,
                    force=200,
                    maxVelocity=3.0
                )
            
            p.stepSimulation()
            #time.sleep(1.0/240.0)
            
            # 每 3 秒印一次狀態
            if step % 720 == 0:
                print(
                    f"Time {step/240:.1f}s: Robot at "
                    f"({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), "
                    f"Distance to cube: {dist:.2f}m"
                )
        
        final_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        final_dist = math.sqrt(
            (final_pos[0] - cube_pos[0])**2 + 
            (final_pos[1] - cube_pos[1])**2
        )
        print(f"\nFinal distance to cube: {final_dist:.2f}m")

def main():
    controller = RobotController()
    
    print("Setting up simulation with red cube target...")
    controller.setup_simulation()
    
    if not controller.load_robot():
        print("Failed to load robot!")
        return
    
    controller.analyze_joints()
    if not controller.controllable_joints:
        print("No controllable joints found!")
        return
    
    print(f"\nRed cube is fixed at position: {controller.cube_position}")
    
    try:
        print("\nChoose option:")
        print("1. Run new optimization")
        print("2. Load saved parameters and demonstrate")
        print("3. Load saved parameters and continue optimization")
        
        choice = input("Enter choice (1, 2, or 3): ").strip()
        
        if choice == "2":
            if controller.load_best_params():
                print("Demonstrating loaded parameters...")
                controller.demonstrate_best_params(duration=20.0)
            else:
                print("Failed to load parameters. Running new optimization instead.")
                choice = "1"
        
        elif choice == "3":
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
            print("Starting new cube-following optimization...")
            generations = int(input("Enter number of generations (default 40): ") or "40")
            best_params = controller.hill_climber_optimization(generations=generations)
            
            controller.plot_learning_curve()
            
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
