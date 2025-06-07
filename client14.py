import pybullet as p
import time
import pybullet_data
import math
import random
import matplotlib.pyplot as plt
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
        self.best_fitness = -float('inf')
        self.best_params = None

        # File for saving/loading參數
        self.params_file = "best_robot_params.json"

        # 定義固定紅色立方體的位置（XYZ）
        self.cube_position = [2.0, 0.0, 0.1]  # 例如放在 x=2.0, y=0.0，高度 0.1 m
        self.cubeId = None

    def setup_simulation(self):
        """Initialize PyBullet simulation environment and load a fixed red cube."""
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")

        # 建立紅色立方體 (side=0.2m)
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.1, 0.1, 0.1],
            rgbaColor=[1, 0, 0, 1]  # 紅色
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.1, 0.1, 0.1]
        )
        # baseMass = 0 → 靜態物件
        self.cubeId = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.cube_position
        )

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
            lower_limit = self.joint_info[joint_idx]['lower_limit']
            upper_limit = self.joint_info[joint_idx]['upper_limit']
            range_size = upper_limit - lower_limit

            offset = random.uniform(lower_limit + 0.1 * range_size,
                                    upper_limit - 0.1 * range_size)
            amplitude = random.uniform(0.1 * range_size, 0.4 * range_size)
            phase = random.uniform(0, 2 * math.pi)

            self.locomotion_params[joint_idx] = [offset, amplitude, phase]
            print(f"Joint {joint_idx} ({self.joint_info[joint_idx]['name']}): offset={offset:.3f}, amp={amplitude:.3f}, phase={phase:.3f}")

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
            params_data = {
                'best_fitness': self.best_fitness,
                'best_params': {str(k): v for k, v in self.best_params.items()},
                'joint_info': {str(k): v for k, v in self.joint_info.items()},
                'learning_history': self.learning_history
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

            print(f"Best parameters loaded from {self.params_file}")
            print(f"Loaded best fitness: {self.best_fitness:.4f}")

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
            print(f"  Best Fitness: {params_data['best_fitness']:.4f}")
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

    def evaluate_locomotion(self, params, duration=10.0, threshold=0.2):
        """
        評估 locomotion 表現（追逐固定紅色立方體）：
        - 將 robot 底座重設到原點，立方體保持在 self.cube_position。
        - 每個 time‐step 都計算「機器人底座中心與紅色立方體間距離 dist_to_cube」。
        - 若 dist_to_cube <= threshold，就視為「追到」，記錄 time_to_reach 並停止模擬。
          fitness = 1.0 / time_to_reach（時間愈短，適應度愈高）。
        - 若跑完所有步數仍未到達，就以「最終距離 dist_to_cube_final」作 fallback：
          fitness = 1.0 / dist_to_cube_final（距離愈近，適應度愈高）。
        """
        # 重置 Robot 底座位置與姿態
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])
        # 先讓機器人穩定一段時間
        for _ in range(100):
            p.stepSimulation()

        # 取初始位置
        initial_robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)

        max_steps = int(duration * 240)  # 240 Hz 模擬
        time_to_reach = None

        for step in range(max_steps):
            t = step * (1.0 / 240.0)

            # 控制各關節
            for joint_idx in self.controllable_joints:
                offset, amplitude, phase = params[joint_idx]
                ll = self.joint_info[joint_idx]['lower_limit']
                ul = self.joint_info[joint_idx]['upper_limit']

                target = offset + amplitude * math.sin(2 * math.pi * t + phase)
                if target < ll: target = ll
                if target > ul: target = ul

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

            # 每個 time‐step 檢查 robot 與立方體的距離
            robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)
            cube_pos, _ = p.getBasePositionAndOrientation(self.cubeId)

            dx = robot_pos[0] - cube_pos[0]
            dy = robot_pos[1] - cube_pos[1]
            dist_to_cube = math.sqrt(dx * dx + dy * dy)

            # 如果距離 <= 閾值，就認為「追到」立方體
            if dist_to_cube <= threshold:
                time_to_reach = t
                break

        # 若在 max_steps 內追到立方體
        if time_to_reach is not None:
            if time_to_reach > 0:
                return 1.0 / time_to_reach
            else:
                return float('inf')

        # 若沒追到，計算最後的距離作 fallback
        final_robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        final_cube_pos, _ = p.getBasePositionAndOrientation(self.cubeId)
        dx = final_robot_pos[0] - final_cube_pos[0]
        dy = final_robot_pos[1] - final_cube_pos[1]
        dist_final = math.sqrt(dx * dx + dy * dy)

        # 距離越近，fitness 越高
        if dist_final > 0:
            return 1.0 / dist_final
        else:
            return float('inf')

    def hill_climber_optimization(self, generations=50, threshold=0.2):
        """
        Hill-climbing 優化 locomotion 參數（追逐紅色立方體）：
        - threshold：距離小於此值就視為「追到」。
        - fitness 計算依據 evaluate_locomotion()。
        """
        print(f"\nStarting hill climber optimization (chase cube) for {generations} generations...")
        self.learning_history = []

        # 1. 隨機初始化參數
        self.initialize_random_params()
        current_params = self.locomotion_params.copy()

        # 2. 初始評估
        current_fitness = self.evaluate_locomotion(current_params, duration=10.0, threshold=threshold)
        self.best_fitness = current_fitness
        self.best_params = current_params.copy()
        self.learning_history.append(current_fitness)
        print(f"Initial fitness: {current_fitness:.4f}")

        # 3. 迭代優化
        for gen in range(generations):
            candidate_params = self.mutate_params(current_params)
            candidate_fitness = self.evaluate_locomotion(candidate_params, duration=10.0, threshold=threshold)

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

        # 存檔
        self.save_best_params()
        return self.best_params

    def plot_learning_curve(self):
        """Plot the learning curve"""
        if not self.learning_history:
            print("No learning history to plot!")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.learning_history, linewidth=2)
        plt.title('Chase Cube Learning Curve')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.grid(True, alpha=0.3)
        plt.show()

        print(f"\nLearning Statistics:")
        print(f"Initial fitness: {self.learning_history[0]:.4f}")
        print(f"Final fitness: {self.learning_history[-1]:.4f}")
        print(f"Best fitness: {max(self.learning_history):.4f}")
        print(f"Improvement: {self.learning_history[-1] - self.learning_history[0]:.4f}")

    def demonstrate_best_params(self, duration=10.0, threshold=0.2):
        """Demonstrate the robot using最佳參數去追立方體"""
        if self.best_params is None:
            print("No optimized parameters found! Run optimization first.")
            return

        print(f"\nDemonstrating best parameters for {duration} seconds (threshold={threshold})...")
        print("Best parameters:")
        for joint_idx, params in self.best_params.items():
            joint_name = self.joint_info[joint_idx]['name']
            print(f"  {joint_name}: offset={params[0]:.3f}, amp={params[1]:.3f}, phase={params[2]:.3f}")

        # Reset robot 底座位置
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])

        steps = int(duration * 240)
        for step in range(steps):
            t = step * (1.0 / 240.0)
            for joint_idx in self.controllable_joints:
                offset, amplitude, phase = self.best_params[joint_idx]
                ll = self.joint_info[joint_idx]['lower_limit']
                ul = self.joint_info[joint_idx]['upper_limit']
                target = offset + amplitude * math.sin(2 * math.pi * t + phase)
                if target < ll: target = ll
                if target > ul: target = ul

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

            # 每隔一段時間印出與立方體的距離
            if step % 480 == 0:  # 每 2 秒
                robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)
                cube_pos, _ = p.getBasePositionAndOrientation(self.cubeId)
                dist = math.sqrt((robot_pos[0]-cube_pos[0])**2 + (robot_pos[1]-cube_pos[1])**2)
                print(f"Time {step/240:.1f}s: Distance to cube = {dist:.3f} m")

def main():
    """Main function to run the complete simulation"""
    controller = RobotController()

    # 1. Setup simulation 並載入紅色立方體
    print("Setting up simulation and loading red cube...")
    controller.setup_simulation()

    # 2. Load robot
    if not controller.load_robot():
        print("Failed to load robot!")
        return

    # 3. Analyze joints
    controller.analyze_joints()
    if not controller.controllable_joints:
        print("No controllable joints found!")
        return

    # 4. 印出是否有已存檔的最佳參數
    controller.print_saved_params_info()

    # 5. 執行 hill‐climber 優化以追逐紅色立方體
    print("\n=== 優化目標：追逐固定紅色立方體 ===")
    best_params = controller.hill_climber_optimization(
        generations=50,
        threshold=0.2  # 當距離 <= 0.2 m 視為「追到」
    )
    controller.plot_learning_curve()
    controller.demonstrate_best_params(duration=10.0, threshold=0.2)

    print("Disconnecting...")
    p.disconnect()

if __name__ == "__main__":
    main()
