import pybullet as p
import pybullet_data
import time
import numpy as np
import random
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
        
        # 紅色方塊固定在 (3, 0, 0.5)
        self.cube_position = [3, 0, 0.5]
        
        # 最佳參數＆紀錄
        self.best_params = None
        self.best_fitness = -float('inf')
        self.params_file = "best_ga_params.json"
        
        # 遺傳演算法相關設定
        self.pop_size = 20             # 族群大小
        self.num_generations = 50      # 代數
        self.num_parents = 5           # 每代保留最優父母數
        self.mutation_rate = 0.2       # 變異率
        
        # 假設每個關節 4 個控制參數: [offset, amplitude, phase, frequency]
        # 總維度 = len(controllable_joints) * 4
        # 下列上下界是示範，請視實際需求微調
        self.lower_bounds = None
        self.upper_bounds = None
        
        # 嘗試載入先前最佳參數
        if os.path.exists(self.params_file):
            with open(self.params_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.best_params = np.array(data["best_params"])
                self.best_fitness = data["best_fitness"]
    
    def setup_simulation(self):
        """
        初始化 PyBullet 模擬環境：載入地板、機器人、紅色方塊。
        """
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        
        # 載入地板
        self.planeId = p.loadURDF("plane.urdf")
        
        # 載入機器人 (請自行替換成你的 robot URDF)
        startPos = [0, 0, 0.5]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robotId = p.loadURDF("your_robot.urdf", startPos, startOrientation)
        
        # 取得可控制關節列表及上下界
        num_joints = p.getNumJoints(self.robotId)
        for ji in range(num_joints):
            info = p.getJointInfo(self.robotId, ji)
            joint_type = info[2]
            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                self.controllable_joints.append(ji)
                # 範例：假設關節限制為 [-1.0, 1.0]
                self.joint_info[ji] = {
                    'lower_limit': -1.0,
                    'upper_limit': 1.0
                }
        
        # 計算參數維度並設定上下界陣列
        dim = len(self.controllable_joints) * 4
        self.lower_bounds = np.tile(np.array([ -0.5, 0.0, 0.0, 0.1 ]), len(self.controllable_joints))
        self.upper_bounds = np.tile(np.array([  0.5, 1.0, 2*np.pi, 2.0 ]), len(self.controllable_joints))
        
        # 載入紅色方塊（使用簡單的 cube 代替示範）
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                              halfExtents=[0.2, 0.2, 0.2],
                                              rgbaColor=[1, 0, 0, 1])
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                    halfExtents=[0.2, 0.2, 0.2])
        self.cubeId = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=collision_shape_id,
                                        baseVisualShapeIndex=visual_shape_id,
                                        basePosition=self.cube_position)
    
    def reset_robot(self):
        """重置機器人到初始位姿 (簡單做法：重新載入或直接歸零所有關節)"""
        # 若要更快速，可只重置位置及關節值
        p.resetBasePositionAndOrientation(self.robotId, [0,0,0.5], p.getQuaternionFromEuler([0,0,0]))
        for ji in self.controllable_joints:
            p.resetJointState(self.robotId, ji, targetValue=0.0)
    
    def evaluate_params(self, params, sim_duration=5.0):
        """
        使用一組參數執行模擬，讓機器人嘗試朝紅色方塊移動。
        回傳最終距離的「負值」作為適應值 (距離越小，適應值越高)。
        """
        # 重置環境
        p.resetSimulation()
        self.setup_simulation()
        self.reset_robot()
        
        # 參數打包：拆成每個關節的子向量
        joint_params = {}
        for idx, ji in enumerate(self.controllable_joints):
            offset = params[idx*4 + 0]
            amplitude = params[idx*4 + 1]
            phase = params[idx*4 + 2]
            freq = params[idx*4 + 3]
            joint_params[ji] = (offset, amplitude, phase, freq)
        
        t0 = time.time()
        while time.time() - t0 < sim_duration:
            t = time.time() - t0
            
            # 針對每個關節計算目標值 (使用正弦波)
            for ji in self.controllable_joints:
                off, amp, ph, fr = joint_params[ji]
                target = off + amp * np.sin(2*np.pi * fr * t + ph)
                # 裁切到關節限制
                low = self.joint_info[ji]['lower_limit']
                high = self.joint_info[ji]['upper_limit']
                target = max(low, min(high, target))
                # 應用 position control
                p.setJointMotorControl2(bodyUniqueId=self.robotId,
                                        jointIndex=ji,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=target,
                                        force=150,
                                        maxVelocity=2.0)
            
            p.stepSimulation()
            # 可以視需要加速模擬：time.sleep(1/240)
        
        # 模擬結束，計算最終機器人與紅方塊距離
        robot_pos, _ = p.getBasePositionAndOrientation(self.robotId)
        dist = np.linalg.norm(np.array(robot_pos) - np.array(self.cube_position))
        
        # 適應值：距離越小越好 → 取負距離
        fitness = -dist
        return fitness
    
    def train_with_ga(self):
        """
        使用遺傳演算法進行參數優化，得到最佳參數並存檔。
        """
        dim = len(self.controllable_joints) * 4
        lb = self.lower_bounds
        ub = self.upper_bounds
        
        # 1. 初始化族群
        population = initialize_population(self.pop_size, dim, lb, ub)
        
        for gen in range(self.num_generations):
            # 2. 計算適應值
            fitnesses = evaluate_population(population, self.evaluate_params)
            
            # 更新全局最佳
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > self.best_fitness:
                self.best_fitness = fitnesses[max_idx]
                self.best_params = population[max_idx].copy()
            
            print(f"第 {gen+1} 代: 最佳適應值 = {fitnesses[max_idx]:.4f}, 全局最佳 = {self.best_fitness:.4f}")
            
            # 3. 選擇父母
            parents = select_parents(population, fitnesses, self.num_parents)
            
            # 4. 生成下一代
            population = create_next_generation(parents,
                                                self.pop_size,
                                                self.mutation_rate,
                                                lb, ub)
        
        # 儲存最佳參數到檔案
        out_data = {
            "best_params": self.best_params.tolist(),
            "best_fitness": float(self.best_fitness)
        }
        with open(self.params_file, 'w', encoding='utf-8') as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        
        print("遺傳演算法訓練結束，最佳適應值：", self.best_fitness)
        return self.best_params

# 補充：將前述 GA 函式複製貼上到此
def initialize_population(pop_size, num_params, lower_bounds, upper_bounds):
    population = []
    for _ in range(pop_size):
        individual = np.random.uniform(lower_bounds, upper_bounds)
        population.append(individual)
    return population

def evaluate_population(population, evaluate_fn):
    fitnesses = []
    for indiv in population:
        fitness = evaluate_fn(indiv)
        fitnesses.append(fitness)
    return np.array(fitnesses)

def select_parents(population, fitnesses, num_parents):
    min_fit = np.min(fitnesses)
    if min_fit < 0:
        fitnesses = fitnesses - min_fit + 1e-6
    total_fit = np.sum(fitnesses)
    probs = fitnesses / total_fit
    parents_idx = np.random.choice(len(population), size=num_parents, p=probs, replace=False)
    parents = [population[i] for i in parents_idx]
    return parents

def crossover(parent1, parent2):
    num_params = len(parent1)
    cx_point = random.randint(1, num_params - 1)
    child1 = np.concatenate([parent1[:cx_point], parent2[cx_point:]])
    child2 = np.concatenate([parent2[:cx_point], parent1[cx_point:]])
    return child1, child2

def mutate(individual, mutation_rate, lower_bounds, upper_bounds):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            span = upper_bounds[i] - lower_bounds[i]
            perturb = np.random.normal(0, 0.1 * span)
            individual[i] += perturb
            if individual[i] < lower_bounds[i]:
                individual[i] = lower_bounds[i]
            if individual[i] > upper_bounds[i]:
                individual[i] = upper_bounds[i]
    return individual

def create_next_generation(parents, pop_size, mutation_rate, lower_bounds, upper_bounds):
    next_pop = []
    next_pop.extend(parents)
    while len(next_pop) < pop_size:
        p1, p2 = random.sample(parents, 2)
        child1, child2 = crossover(p1, p2)
        child1 = mutate(child1, mutation_rate, lower_bounds, upper_bounds)
        child2 = mutate(child2, mutation_rate, lower_bounds, upper_bounds)
        next_pop.append(child1)
        if len(next_pop) < pop_size:
            next_pop.append(child2)
    return next_pop[:pop_size]

# ====== 主程式範例 ======
if __name__ == "__main__":
    controller = RobotController()
    controller.setup_simulation()
    # 開始遺傳演算法訓練
    best = controller.train_with_ga()
    print("訓練完成，最佳參數：", best)
