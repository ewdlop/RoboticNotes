import pybullet as p
import time
import pybullet_data
import math
import random
import matplotlib.pyplot as plt
import json
import os
import numpy as np

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

        # 檔案名稱：存放最佳參數
        self.params_file = "best_robot_params.json"

        # 定義固定紅色立方體的位置（XYZ），高度0.1m
        self.cube_position = [2.0, 0.0, 0.1]
        self.cubeId = None

        # Evolutionary Algorithm parameters
        self.population_size = 20
        self.elite_size = 4
        self.mutation_rate = 0.15
        self.crossover_rate = 0.7

    def setup_simulation(self):
        """Initialize PyBullet simulation environment and load a fixed red cube."""
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")

        # 建立紅色立方體 (邊長=0.2m)
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.1, 0.1, 0.1],
            rgbaColor=[1, 0, 0, 1]  # 紅色
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.1, 0.1, 0.1]
        )
        # baseMass=0 → 靜態物件
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
            print(f"MJCF 載入成功，Bodies 數量：{len(robotIds)}")

            if len(robotIds) > 1:
                self.robotId = robotIds[1]
            else:
                self.robotId = robotIds[0]

            print(f"使用 robotId：{self.robotId}")

            # 設定初始位置與姿態
            p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])
            return True

        except Exception as e:
            print(f"MJCF 載入失敗：{e}")
            return False

    def analyze_joints(self):
        """Analyze and categorize all joints in the robot"""
        numJoints = p.getNumJoints(self.robotId)
        print(f"\n機器人共有 {numJoints} 個關節")

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

            # 跳過 root 關節，只記錄有下限與上限的鉸鏈關節
            if jointType == 0 and jointName != 'root':
                if jointLowerLimit < jointUpperLimit:
                    self.controllable_joints.append(i)
                    print(f"可控制關節 {i}：{jointName}")

        print(f"總計可控制關節：{len(self.controllable_joints)}")
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
            print(f"關節 {joint_idx}（{self.joint_info[joint_idx]['name']}）：offset={offset:.3f}, amp={amplitude:.3f}, phase={phase:.3f}")

    def create_random_individual(self):
        """Create a single random individual (parameter set) for evolutionary algorithm"""
        individual = {}
        for joint_idx in self.controllable_joints:
            lower_limit = self.joint_info[joint_idx]['lower_limit']
            upper_limit = self.joint_info[joint_idx]['upper_limit']
            range_size = upper_limit - lower_limit

            offset = random.uniform(lower_limit + 0.1 * range_size,
                                    upper_limit - 0.1 * range_size)
            amplitude = random.uniform(0.1 * range_size, 0.4 * range_size)
            phase = random.uniform(0, 2 * math.pi)

            individual[joint_idx] = [offset, amplitude, phase]
        return individual

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

    def crossover_params(self, parent1, parent2):
        """Create offspring by crossing over two parent parameter sets"""
        child = {}
        for joint_idx in parent1:
            if random.random() < self.crossover_rate:
                # Arithmetic crossover for each parameter
                alpha = random.random()
                child[joint_idx] = [
                    alpha * parent1[joint_idx][0] + (1 - alpha) * parent2[joint_idx][0],  # offset
                    alpha * parent1[joint_idx][1] + (1 - alpha) * parent2[joint_idx][1],  # amplitude
                    alpha * parent1[joint_idx][2] + (1 - alpha) * parent2[joint_idx][2]   # phase
                ]
                
                # Ensure constraints
                lower_limit = self.joint_info[joint_idx]['lower_limit']
                upper_limit = self.joint_info[joint_idx]['upper_limit']
                range_size = upper_limit - lower_limit
                
                # Clamp offset
                child[joint_idx][0] = max(lower_limit, min(upper_limit, child[joint_idx][0]))
                # Clamp amplitude
                child[joint_idx][1] = max(0, min(0.5 * range_size, child[joint_idx][1]))
                # Wrap phase
                child[joint_idx][2] = child[joint_idx][2] % (2 * math.pi)
            else:
                # Copy from one parent randomly
                if random.random() < 0.5:
                    child[joint_idx] = parent1[joint_idx].copy()
                else:
                    child[joint_idx] = parent2[joint_idx].copy()
        return child

    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Tournament selection for choosing parents"""
        tournament_indices = random.sample(range(len(population)), 
                                          min(tournament_size, len(population)))
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()

    def save_best_params(self):
        """Save the best parameters to a JSON file"""
        if self.best_params is None:
            print("沒有最佳參數可存檔！")
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

            print(f"最佳參數已存到 {self.params_file}")
            print(f"最佳適應度：{self.best_fitness:.4f}")
            return True

        except Exception as e:
            print(f"存檔失敗：{e}")
            return False

    def load_best_params(self):
        """Load the best parameters from a JSON file"""
        if not os.path.exists(self.params_file):
            print(f"找不到已存檔的參數：{self.params_file}")
            return False

        try:
            with open(self.params_file, 'r') as f:
                params_data = json.load(f)

            self.best_fitness = params_data['best_fitness']
            self.best_params = {int(k): v for k, v in params_data['best_params'].items()}
            self.learning_history = params_data.get('learning_history', [])

            print(f"成功從 {self.params_file} 載入最佳參數")
            print(f"載入的最佳適應度：{self.best_fitness:.4f}")

            if self.joint_info:
                loaded_joints = set(self.best_params.keys())
                current_joints = set(self.controllable_joints)
                if loaded_joints != current_joints:
                    print("警告：載入參數與目前機器人關節不符！")
                    print(f"載入的關節：{loaded_joints}")
                    print(f"目前的關節：{current_joints}")
                    return False

            return True

        except Exception as e:
            print(f"載入失敗：{e}")
            return False

    def print_saved_params_info(self):
        """Print information about saved parameters without loading them"""
        if not os.path.exists(self.params_file):
            print(f"找不到已存檔的參數：{self.params_file}")
            return

        try:
            with open(self.params_file, 'r') as f:
                params_data = json.load(f)

            print(f"\n=== 已存參數資訊 ===")
            print(f"  檔案：{self.params_file}")
            print(f"  最佳適應度：{params_data['best_fitness']:.4f}")
            print(f"  關節數量：{len(params_data['best_params'])}")
            print(f"  學習歷史長度：{len(params_data.get('learning_history', []))}")

            if 'best_params' in params_data:
                print("  各關節參數：")
                joint_info = params_data.get('joint_info', {})
                for joint_idx_str, params in params_data['best_params'].items():
                    joint_name = joint_info.get(joint_idx_str, {}).get('name', f'Joint_{joint_idx_str}')
                    print(f"    {joint_name}: offset={params[0]:.3f}, amp={params[1]:.3f}, phase={params[2]:.3f}")

        except Exception as e:
            print(f"讀取參數失敗：{e}")

    def evaluate_locomotion(self, params, duration=10.0, threshold=0.2):
        """
        評估 locomotion 表現（追逐固定紅色立方體）：
        - 重置 robot 底座到 [0,0,1]，立方體保持在 self.cube_position。
        - 每個 time-step 都計算「機器人底座中心與紅色立方體間距離 dist_to_cube」。
        - 若 dist_to_cube <= threshold，就視為「追到」，時間記為 time_to_reach。
          fitness = 1.0 / time_to_reach（時間愈短適應度愈高）。
        - 若跑完所有步數仍未到，則以「最終距離 dist_to_cube_final」作 fallback：
          fitness = 1.0 / dist_to_cube_final（距離愈近適應度愈高）。
        """
        # 重置 Robot 底座位置與姿態
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])
        # 先讓機器人穩定一段時間
        for _ in range(100):
            p.stepSimulation()

        max_steps = int(duration * 240)  # 240 Hz 模擬頻率
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

            # 每個 time-step 檢查 robot 與立方體的距離
            robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)
            cube_pos, _ = p.getBasePositionAndOrientation(self.cubeId)

            dx = robot_pos[0] - cube_pos[0]
            dy = robot_pos[1] - cube_pos[1]
            dist_to_cube = math.sqrt(dx * dx + dy * dy)

            # 如果距離 <= 閾值，就認為「追到」立方體
            if dist_to_cube <= threshold:
                time_to_reach = t
                break

        # 若在 max_steps 內追到
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
        print(f"\n開始 hill climber 優化（追立方體），共 {generations} 代...")
        self.learning_history = []

        # 1. 隨機初始化參數
        self.initialize_random_params()
        current_params = self.locomotion_params.copy()

        # 2. 初始評估
        current_fitness = self.evaluate_locomotion(current_params, duration=10.0, threshold=threshold)
        self.best_fitness = current_fitness
        self.best_params = current_params.copy()
        self.learning_history.append(current_fitness)
        print(f"初始適應度：{current_fitness:.4f}")

        # 3. 迭代優化
        for gen in range(generations):
            candidate_params = self.mutate_params(current_params)
            candidate_fitness = self.evaluate_locomotion(candidate_params, duration=10.0, threshold=threshold)

            if candidate_fitness > current_fitness:
                current_params = candidate_params
                current_fitness = candidate_fitness
                print(f"第 {gen+1} 代：新最佳適應度 = {current_fitness:.4f}")
                if current_fitness > self.best_fitness:
                    self.best_fitness = current_fitness
                    self.best_params = current_params.copy()
            else:
                print(f"第 {gen+1} 代：適應度 = {candidate_fitness:.4f}（無改進）")

            self.learning_history.append(current_fitness)

        print(f"\n優化結束！")
        print(f"最佳適應度：{self.best_fitness:.4f}")

        # 存檔
        self.save_best_params()
        return self.best_params

    def evolutionary_algorithm_optimization(self, generations=50, threshold=0.2, 
                                          population_size=20, elite_size=4, 
                                          mutation_rate=0.15, crossover_rate=0.7):
        """
        Evolutionary Algorithm 優化 locomotion 參數（追逐紅色立方體）：
        使用族群演化、選擇、交配、突變等機制
        """
        print(f"\n開始演化算法優化（追立方體），共 {generations} 代...")
        print(f"族群大小：{population_size}, 精英保留：{elite_size}")
        print(f"突變率：{mutation_rate}, 交配率：{crossover_rate}")
        
        self.learning_history = []
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # 1. 初始化族群
        print("初始化族群...")
        population = []
        fitness_scores = []
        
        for i in range(population_size):
            individual = self.create_random_individual()
            population.append(individual)
            fitness = self.evaluate_locomotion(individual, duration=8.0, threshold=threshold)
            fitness_scores.append(fitness)
            print(f"個體 {i+1}/{population_size}: 適應度 = {fitness:.4f}")

        # 找到初始最佳個體
        best_idx = fitness_scores.index(max(fitness_scores))
        self.best_fitness = fitness_scores[best_idx]
        self.best_params = population[best_idx].copy()
        self.learning_history.append(self.best_fitness)
        
        print(f"初始族群最佳適應度：{self.best_fitness:.4f}")

        # 2. 演化循環
        for gen in range(generations):
            print(f"\n=== 第 {gen+1} 代 ===")
            
            # 評估整個族群
            fitness_scores = []
            for individual in population:
                fitness = self.evaluate_locomotion(individual, duration=8.0, threshold=threshold)
                fitness_scores.append(fitness)

            # 更新全域最佳
            current_best_idx = fitness_scores.index(max(fitness_scores))
            current_best_fitness = fitness_scores[current_best_idx]
            
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_params = population[current_best_idx].copy()
                print(f"發現新最佳！適應度：{self.best_fitness:.4f}")
            
            # 記錄當代最佳適應度
            self.learning_history.append(current_best_fitness)
            
            # 統計資訊
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            print(f"當代最佳：{current_best_fitness:.4f}, 平均：{avg_fitness:.4f}")

            # 3. 選擇 (Tournament Selection)
            new_population = []
            
            # 精英保留：直接複製最佳個體
            sorted_indices = sorted(range(len(fitness_scores)), 
                                   key=lambda i: fitness_scores[i], reverse=True)
            for i in range(elite_size):
                new_population.append(population[sorted_indices[i]].copy())
            
            # 生成剩餘個體
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                # 交配產生子代
                if random.random() < crossover_rate:
                    child = self.crossover_params(parent1, parent2)
                else:
                    child = parent1.copy() if random.random() < 0.5 else parent2.copy()
                
                # 突變
                child = self.mutate_params(child, mutation_rate)
                new_population.append(child)

            population = new_population

        print(f"\n演化算法優化結束！")
        print(f"最佳適應度：{self.best_fitness:.4f}")

        # 存檔
        self.save_best_params()
        return self.best_params

    def plot_learning_curve(self):
        """Plot the learning curve (Fitness 的單位為 1/秒 或 1/公尺)"""
        if not self.learning_history:
            print("No learning history to plot!")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.learning_history, linewidth=2)
        plt.title('Chase Cube Learning Curve')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.grid(True, alpha=0.3)
        plt.show()

        print(f"\nLearning Statistics:")
        print(f"Initial fitness: {self.learning_history[0]:.4f}")
        print(f"Final fitness:   {self.learning_history[-1]:.4f}")
        print(f"Best fitness:    {max(self.learning_history):.4f}")
        print(f"Improvement:     {self.learning_history[-1] - self.learning_history[0]:.4f}")

    def demonstrate_best_params(self, duration=10.0, threshold=0.2):
        """
        Demonstrate the robot using 最佳參數去追立方體，並計算平均速度 (cm/s)：
        - 先重置位置，再依最佳參數執行一段時間的追逐，
        - 同時記錄底座位移，最後計算總路徑長度 / 時間 = 平均速度 (m/s)，
          再乘以100得到 cm/s。
        """
        if self.best_params is None:
            print("沒有最佳參數！請先執行優化或載入參數。")
            return

        print(f"\n示範最佳參數（共 {duration} 秒，threshold={threshold}）：")
        print("各關節最佳參數：")
        for joint_idx, params in self.best_params.items():
            joint_name = self.joint_info[joint_idx]['name']
            print(f"  {joint_name}：offset={params[0]:.3f}, amp={params[1]:.3f}, phase={params[2]:.3f}")

        # 重置機器人底座位置
        p.resetBasePositionAndOrientation(self.robotId, [0, 0, 1], [0, 0, 0, 1])
        # 多紀錄每個 time-step 的底座位置，用於計算路徑長度
        prev_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        total_distance = 0.0  # m
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

            # 計算這個 time-step 走過的水平距離（m），累加
            curr_pos, _ = p.getBasePositionAndOrientation(self.robotId)
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            total_distance += dist
            prev_pos = curr_pos

            # 每 2 秒印出一次與立方體的距離
            if step % 480 == 0:
                robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)
                cube_pos, _ = p.getBasePositionAndOrientation(self.cubeId)
                d2c = math.sqrt((robot_pos[0]-cube_pos[0])**2 + (robot_pos[1]-cube_pos[1])**2)
                print(f"Time {step/240:.1f}s：距離立方體 = {d2c:.3f} m")

        # 計算平均速度 (m/s) → 轉為 cm/s
        avg_speed_m_s = total_distance / duration if duration > 0 else 0.0
        avg_speed_cm_s = avg_speed_m_s * 100.0
        print(f"\n示範結束！")
        print(f"總行走水平距離：{total_distance:.3f} m")
        print(f"平均速度：{avg_speed_cm_s:.2f} cm/s")

def main():
    """Main function: 提供「訓練或僅示範」的模式選擇，包含演化算法選項"""
    controller = RobotController()

    # 1. 設定模擬環境並載入紅色立方體
    print("設定模擬環境並載入紅色立方體...")
    controller.setup_simulation()

    # 2. 載入機器人
    if not controller.load_robot():
        print("機器人載入失敗，程式結束")
        return

    # 3. 分析並紀錄可控制關節
    controller.analyze_joints()
    if not controller.controllable_joints:
        print("找不到可控制關節，程式結束")
        return

    # 4. 印出已存參數摘要
    controller.print_saved_params_info()

    # 5. 讓使用者選擇模式：1.Hill-Climber 2.Evolutionary Algorithm 3.載入並示範
    print("\n請選擇模式：")
    print("  1. Hill-Climber 訓練並示範")
    print("  2. Evolutionary Algorithm 訓練並示範")
    print("  3. 僅載入已存的最佳參數並示範 (非訓練模式)")
    choice = input("輸入數字 (1, 2 或 3)：").strip()

    if choice == '1':
        # 執行 hill-climber 優化並示範
        print("\n=== 模式：Hill-Climber 訓練並示範 ===")
        best_params = controller.hill_climber_optimization(
            generations=100,
            threshold=0.2  # 當距離 <= 0.2 m 視為「追到」
        )
        controller.plot_learning_curve()
        controller.demonstrate_best_params(duration=10.0, threshold=0.2)

    elif choice == '2':
        # 執行 evolutionary algorithm 優化並示範
        print("\n=== 模式：Evolutionary Algorithm 訓練並示範 ===")
        
        # 讓使用者自訂演化算法參數
        print("\n演化算法參數設定（按 Enter 使用預設值）：")
        
        try:
            pop_size = input(f"族群大小 (預設 20): ").strip()
            pop_size = int(pop_size) if pop_size else 20
            
            generations = input(f"演化代數 (預設 50): ").strip()
            generations = int(generations) if generations else 50
            
            elite_size = input(f"精英保留數量 (預設 4): ").strip()
            elite_size = int(elite_size) if elite_size else 4
            
            mutation_rate = input(f"突變率 (預設 0.15): ").strip()
            mutation_rate = float(mutation_rate) if mutation_rate else 0.15
            
            crossover_rate = input(f"交配率 (預設 0.7): ").strip()
            crossover_rate = float(crossover_rate) if crossover_rate else 0.7
            
        except ValueError:
            print("輸入格式錯誤，使用預設參數")
            pop_size, generations, elite_size = 20, 50, 4
            mutation_rate, crossover_rate = 0.15, 0.7
        
        print(f"\n使用參數：族群={pop_size}, 代數={generations}, 精英={elite_size}")
        print(f"突變率={mutation_rate}, 交配率={crossover_rate}")
        
        best_params = controller.evolutionary_algorithm_optimization(
            generations=generations,
            threshold=0.2,
            population_size=pop_size,
            elite_size=elite_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate
        )
        controller.plot_learning_curve()
        controller.demonstrate_best_params(duration=10.0, threshold=0.2)

    elif choice == '3':
        # 僅載入已存的最佳參數並示範
        print("\n=== 模式：僅載入並示範 ===")
        if controller.load_best_params():
            controller.demonstrate_best_params(duration=10.0, threshold=0.2)
        else:
            print("載入已存參數失敗，無法示範")

    else:
        print("不正確的選擇，程式結束")

    print("結束並斷開連線...")
    p.disconnect()

if __name__ == "__main__":
    main()