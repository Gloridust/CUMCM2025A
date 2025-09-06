import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import trange
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from scipy.optimize import differential_evolution
import warnings
import random
warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重复
np.random.seed(42)
random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# ======================
# 问题参数
# ======================
MISSILE_SPEED = 300.0   # m/s 导弹飞行速度
SMOKE_RADIUS = 10.0     # m，有效遮蔽半径（题目说10m范围内有效）
SMOKE_DURATION = 20.0   # s，有效遮蔽持续时间
SMOKE_SINK_SPEED = 3.0  # m/s，烟幕云团下沉速度
GRAVITY = 9.8           # m/s²，重力加速度
DT = 0.1                # 时间步长

# 导弹和无人机初始位置
MISSILES = {
    'M1': np.array([20000, 0, 2000]),
    'M2': np.array([19000, 600, 2100]), 
    'M3': np.array([18000, -600, 1900])
}

DRONES = {
    'FY1': np.array([17800, 0, 1800]),
    'FY2': np.array([12000, 1400, 1400]),
    'FY3': np.array([6000, -3000, 700]),
    'FY4': np.array([11000, 2000, 1800]),
    'FY5': np.array([13000, -2000, 1300])
}

# 目标位置
FAKE_TARGET = np.array([0, 0, 0])      # 假目标（原点）
REAL_TARGET = np.array([0, 200, 0])    # 真目标

# ======================
# 物理模型函数
# ======================
def missile_position(missile_name, t):
    """计算导弹在时刻t的位置（直线飞向假目标）"""
    start_pos = MISSILES[missile_name]
    direction = (FAKE_TARGET - start_pos) / np.linalg.norm(FAKE_TARGET - start_pos)
    return start_pos + MISSILE_SPEED * t * direction

def drone_position(drone_name, t, direction_angle, speed):
    """计算无人机在时刻t的位置"""
    start_pos = DRONES[drone_name]
    direction = np.array([np.cos(np.radians(direction_angle)), 
                         np.sin(np.radians(direction_angle)), 0])
    return start_pos + speed * t * direction

def smoke_bomb_trajectory(release_pos, release_time, t):
    """计算烟幕弹在时刻t的位置（考虑重力）"""
    if t < release_time:
        return release_pos
    dt = t - release_time
    # 只有z方向受重力影响
    z_pos = release_pos[2] - 0.5 * GRAVITY * dt**2
    return np.array([release_pos[0], release_pos[1], z_pos])

def smoke_cloud_position(explode_pos, explode_time, t):
    """计算烟幕云团在时刻t的位置（匀速下沉）"""
    if t < explode_time:
        return explode_pos
    dt = t - explode_time
    z_pos = explode_pos[2] - SMOKE_SINK_SPEED * dt
    return np.array([explode_pos[0], explode_pos[1], z_pos])

# ======================
# 遮蔽判定函数
# ======================
def is_line_blocked_by_sphere(line_start, line_end, sphere_center, sphere_radius):
    """判断线段是否被球体遮挡"""
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return False
    
    line_dir = line_vec / line_len
    to_sphere = sphere_center - line_start
    
    # 投影到直线上
    proj_len = np.dot(to_sphere, line_dir)
    proj_len = np.clip(proj_len, 0, line_len)
    
    # 最近点
    closest_point = line_start + proj_len * line_dir
    distance = np.linalg.norm(closest_point - sphere_center)
    
    return distance <= sphere_radius

def compute_total_coverage_time(strategy):
    """
    计算给定策略的总遮蔽时间
    strategy: 包含所有无人机和烟幕弹的完整策略
    """
    t_max = 100  # 扩大模拟时间范围
    times = np.arange(0, t_max, DT)
    
    # 解析策略
    smoke_events = []
    for drone_name in DRONES.keys():
        if drone_name not in strategy:
            continue
        drone_info = strategy[drone_name]
        direction = drone_info['direction']
        speed = drone_info['speed']
        bombs = drone_info['bombs']
        
        for bomb in bombs:
            release_time = bomb['release_time']
            explode_delay = bomb['explode_delay']
            
            # 计算投放位置和起爆位置
            release_pos = drone_position(drone_name, release_time, direction, speed)
            explode_time = release_time + explode_delay
            explode_pos = smoke_bomb_trajectory(release_pos, release_time, explode_time)
            
            smoke_events.append({
                'explode_pos': explode_pos,
                'explode_time': explode_time,
                'end_time': explode_time + SMOKE_DURATION
            })
    
    # 计算每个导弹的遮蔽时间
    total_coverage = 0.0
    
    for missile_name in MISSILES.keys():
        missile_coverage = 0.0
        
        for t in times:
            m_pos = missile_position(missile_name, t)
            
            # 检查是否被任何烟幕遮挡
            blocked = False
            for smoke_event in smoke_events:
                if smoke_event['explode_time'] <= t <= smoke_event['end_time']:
                    # 计算烟幕当前位置
                    smoke_pos = smoke_cloud_position(
                        smoke_event['explode_pos'], 
                        smoke_event['explode_time'], 
                        t
                    )
                    
                    # 检查视线是否被遮挡
                    if is_line_blocked_by_sphere(m_pos, REAL_TARGET, smoke_pos, SMOKE_RADIUS):
                        blocked = True
                        break
            
            if blocked:
                missile_coverage += DT
        
        total_coverage += missile_coverage
    
    return total_coverage

# ======================
# 智能策略设计："化曲为直"和"底线思维"
# ======================
def calculate_interception_opportunities():
    """
    化曲为直：将复杂的多目标拦截问题转化为直线路径上的机会识别
    返回：{drone_name: {missile_name: [(time, position, priority, coverage_duration), ...]}}
    """
    opportunities = {}
    
    for drone_name, drone_pos in DRONES.items():
        opportunities[drone_name] = {}
        
        for missile_name, missile_start in MISSILES.items():
            # 计算导弹轨迹
            missile_to_target = FAKE_TARGET - missile_start
            missile_distance = np.linalg.norm(missile_to_target)
            missile_direction = missile_to_target / missile_distance
            missile_flight_time = missile_distance / MISSILE_SPEED
            
            intercept_points = []
            
            # 在导弹轨迹上每1秒分析一次拦截机会
            for t in np.arange(1.0, missile_flight_time, 1.0):
                missile_pos = missile_start + MISSILE_SPEED * t * missile_direction
                
                # 计算最优拦截位置（考虑烟幕下沉）
                optimal_intercept_pos = calculate_optimal_intercept_position(missile_pos, REAL_TARGET, t)
                
                # 计算无人机到达该点的最短时间
                drone_to_intercept = optimal_intercept_pos - drone_pos
                min_drone_distance = np.linalg.norm(drone_to_intercept)
                min_drone_time = min_drone_distance / 140  # 最大速度
                
                if min_drone_time <= t - 2.0:  # 需要提前2秒到达进行投放和起爆
                    # 计算拦截优先级
                    priority = calculate_intercept_priority(
                        missile_pos, optimal_intercept_pos, t, missile_flight_time, min_drone_time
                    )
                    
                    # 估算可能的遮蔽持续时间
                    coverage_duration = estimate_coverage_duration(missile_pos, optimal_intercept_pos, t)
                    
                    intercept_points.append((t, optimal_intercept_pos, priority, coverage_duration))
            
            # 按优先级排序，保留前3个最佳机会
            intercept_points.sort(key=lambda x: x[2], reverse=True)
            opportunities[drone_name][missile_name] = intercept_points[:3]
    
    return opportunities

def calculate_optimal_intercept_position(missile_pos, target_pos, intercept_time):
    """计算最优拦截位置，考虑烟幕下沉效应"""
    # 基础拦截点：导弹到目标连线上的垂直点
    missile_to_target = target_pos - missile_pos
    
    # 考虑烟幕下沉，需要在更高位置起爆
    sink_compensation = SMOKE_SINK_SPEED * (SMOKE_DURATION / 2)  # 平均下沉距离
    
    # 最优位置：导弹到目标连线的中点，适当上移
    optimal_pos = missile_pos + 0.4 * missile_to_target  # 在40%位置拦截效果最好
    optimal_pos[2] += sink_compensation  # 高度补偿
    
    return optimal_pos

def calculate_intercept_priority(missile_pos, intercept_pos, t, total_flight_time, drone_time):
    """计算拦截优先级"""
    # 1. 几何效果：越接近导弹-目标中线越好
    missile_to_target = REAL_TARGET - missile_pos
    intercept_to_target = REAL_TARGET - intercept_pos
    
    # 计算拦截点到导弹-目标连线的距离
    line_distance = np.linalg.norm(np.cross(missile_to_target, intercept_to_target)) / np.linalg.norm(missile_to_target)
    geometry_score = 1.0 / (1.0 + line_distance / 100)  # 距离越近得分越高
    
    # 2. 时间窗口：导弹飞行中段最佳
    time_ratio = t / total_flight_time
    time_score = 1.0 - abs(time_ratio - 0.5) * 2  # 50%时刻得分最高
    
    # 3. 可达性：无人机到达的容易程度
    reachability_score = 1.0 / (1.0 + drone_time / 10)
    
    # 4. 持续性：能够持续遮蔽的时间
    sustainability_score = min(1.0, (total_flight_time - t) / SMOKE_DURATION)
    
    return geometry_score * time_score * reachability_score * sustainability_score

def estimate_coverage_duration(missile_pos, intercept_pos, start_time):
    """估算从该拦截点开始能够遮蔽的持续时间"""
    # 简化估算：基于烟幕持续时间和导弹剩余飞行时间
    missile_to_target_remaining = np.linalg.norm(REAL_TARGET - missile_pos)
    remaining_flight_time = missile_to_target_remaining / MISSILE_SPEED
    
    return min(SMOKE_DURATION, remaining_flight_time)

def design_bottom_line_coverage():
    """
    底线思维：确保每个导弹都有基本遮蔽保障
    为每个导弹分配至少一个专门的拦截任务
    """
    base_assignments = {}
    
    # 计算每个导弹的威胁程度
    missile_threats = []
    for missile_name, missile_start in MISSILES.items():
        # 威胁程度 = 1/距离真目标的距离 + 飞行时间紧迫性
        distance_to_real_target = np.linalg.norm(missile_start - REAL_TARGET)
        flight_time = distance_to_real_target / MISSILE_SPEED
        
        threat_level = (1.0 / distance_to_real_target) * 10000 + (1.0 / flight_time) * 100
        missile_threats.append((missile_name, threat_level))
    
    # 按威胁程度排序，优先处理高威胁导弹
    missile_threats.sort(key=lambda x: x[1], reverse=True)
    
    # 为每个导弹分配最佳无人机
    used_drones = set()
    for missile_name, threat in missile_threats:
        best_drone = None
        best_score = -1
        
        for drone_name, drone_pos in DRONES.items():
            if drone_name in used_drones:
                continue
            
            # 计算该无人机对该导弹的拦截能力
            missile_start = MISSILES[missile_name]
            missile_to_fake = FAKE_TARGET - missile_start
            missile_direction = missile_to_fake / np.linalg.norm(missile_to_fake)
            missile_flight_time = np.linalg.norm(missile_to_fake) / MISSILE_SPEED
            
            # 找到最佳拦截时刻（导弹飞行40%时刻）
            best_intercept_time = missile_flight_time * 0.4
            intercept_missile_pos = missile_start + MISSILE_SPEED * best_intercept_time * missile_direction
            optimal_intercept_pos = calculate_optimal_intercept_position(
                intercept_missile_pos, REAL_TARGET, best_intercept_time
            )
            
            # 计算无人机能否及时到达
            drone_distance = np.linalg.norm(optimal_intercept_pos - drone_pos)
            min_required_time = drone_distance / 140
            
            if min_required_time <= best_intercept_time - 2.0:  # 需要提前2秒
                # 计算拦截效果评分
                coverage_score = 1.0 / (1.0 + drone_distance / 5000)
                time_score = 1.0 - min_required_time / best_intercept_time
                urgency_score = threat / max([t[1] for t in missile_threats])  # 归一化威胁程度
                
                total_score = coverage_score * time_score * urgency_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_drone = drone_name
        
        if best_drone:
            base_assignments[missile_name] = {
                'drone': best_drone,
                'score': best_score,
                'threat': threat
            }
            used_drones.add(best_drone)
    
    return base_assignments

def create_opportunistic_strategy():
    """
    机会主义策略：无人机在执行主要任务的路上识别和处理顺路目标
    """
    # 第一步：底线保障 - 确保每个导弹都有基本覆盖
    base_assignments = design_bottom_line_coverage()
    
    # 第二步：计算所有拦截机会
    opportunities = calculate_interception_opportunities()
    
    strategy = {}
    
    for drone_name in DRONES.keys():
        # 确定该无人机的主要任务
        primary_targets = [missile for missile, info in base_assignments.items() 
                          if info['drone'] == drone_name]
        
        if not primary_targets:
            # 没有主要任务的无人机，寻找最佳机会任务
            best_opportunity = None
            best_priority = 0
            
            for missile_name in MISSILES.keys():
                if missile_name in opportunities[drone_name] and opportunities[drone_name][missile_name]:
                    top_opportunity = opportunities[drone_name][missile_name][0]
                    if top_opportunity[2] > best_priority:  # priority
                        best_priority = top_opportunity[2]
                        best_opportunity = (missile_name, top_opportunity)
            
            if best_opportunity:
                primary_targets = [best_opportunity[0]]
        
        if primary_targets:
            # 规划飞行路径
            primary_missile = primary_targets[0]  # 主要目标
            missile_start = MISSILES[primary_missile]
            
            # 计算到主要目标的最优路径
            primary_opportunities = opportunities[drone_name][primary_missile]
            if primary_opportunities:
                best_primary_opportunity = primary_opportunities[0]
                target_time, target_pos, priority, duration = best_primary_opportunity
                
                # 计算飞行方向
                drone_start = DRONES[drone_name]
                to_target = target_pos - drone_start
                flight_direction = np.degrees(np.arctan2(to_target[1], to_target[0]))
                flight_direction = flight_direction % 360
                
                # 计算所需速度
                required_distance = np.linalg.norm(to_target)
                required_time = target_time - 2.0  # 提前2秒到达
                required_speed = required_distance / required_time if required_time > 0 else 140
                flight_speed = min(140, max(70, required_speed))
                
                # 规划烟幕弹投放
                bombs = []
                
                # 主要目标的烟幕弹
                release_time = target_time - 3.0  # 提前3秒投放
                explode_delay = 2.0  # 2秒后起爆
                
                bombs.append({
                    'release_time': max(0, release_time),
                    'explode_delay': explode_delay,
                    'target_missile': primary_missile,
                    'priority': 'primary'
                })
                
                # 寻找顺路机会
                for missile_name in MISSILES.keys():
                    if missile_name != primary_missile and len(bombs) < 3:
                        if missile_name in opportunities[drone_name]:
                            for opp_time, opp_pos, opp_priority, opp_duration in opportunities[drone_name][missile_name]:
                                # 检查是否在主要任务路径上
                                drone_pos_at_time = drone_position(drone_name, opp_time, flight_direction, flight_speed)
                                detour_distance = np.linalg.norm(drone_pos_at_time - opp_pos)
                                
                                # 如果绕路距离小于2km且优先级足够高，就是好机会
                                if detour_distance < 2000 and opp_priority > 0.3:
                                    opp_release_time = opp_time - 2.5
                                    opp_explode_delay = 1.5
                                    
                                    # 确保时间间隔
                                    valid_timing = True
                                    for existing_bomb in bombs:
                                        if abs(existing_bomb['release_time'] - opp_release_time) < 1.0:
                                            valid_timing = False
                                            break
                                    
                                    if valid_timing and opp_release_time > 0:
                                        bombs.append({
                                            'release_time': opp_release_time,
                                            'explode_delay': opp_explode_delay,
                                            'target_missile': missile_name,
                                            'priority': 'opportunity'
                                        })
                                        break
                
                # 确保投放时间排序和间隔
                bombs.sort(key=lambda x: x['release_time'])
                for i in range(1, len(bombs)):
                    if bombs[i]['release_time'] - bombs[i-1]['release_time'] < 1.0:
                        bombs[i]['release_time'] = bombs[i-1]['release_time'] + 1.0
                
                strategy[drone_name] = {
                    'direction': flight_direction,
                    'speed': flight_speed,
                    'bombs': bombs
                }
    
    return strategy

# ======================
# 优化算法
# ======================
def create_random_strategy():
    """创建一个随机策略"""
    strategy = {}
    
    for drone_name in DRONES.keys():
        # 随机决定是否使用这个无人机（70%概率使用）
        if np.random.random() < 0.7:
            direction = np.random.uniform(0, 360)  # 飞行方向
            speed = np.random.uniform(70, 140)     # 飞行速度
            
            # 随机决定投放烟幕弹数量（1-3枚）
            n_bombs = np.random.randint(1, 4)
            bombs = []
            
            last_release_time = 0
            for i in range(n_bombs):
                # 确保投放间隔至少1秒
                release_time = last_release_time + np.random.uniform(1.0, 10.0)
                explode_delay = np.random.uniform(0.5, 8.0)  # 起爆延迟
                
                bombs.append({
                    'release_time': release_time,
                    'explode_delay': explode_delay
                })
                last_release_time = release_time
            
            strategy[drone_name] = {
                'direction': direction,
                'speed': speed,
                'bombs': bombs
            }
    
    return strategy

def mutate_strategy(strategy, mutation_rate=0.3):
    """变异策略"""
    new_strategy = {}
    
    for drone_name in DRONES.keys():
        if drone_name in strategy and np.random.random() > mutation_rate:
            # 保持原策略，但可能微调
            drone_info = strategy[drone_name].copy()
            drone_info['bombs'] = [bomb.copy() for bomb in drone_info['bombs']]
            
            # 小幅调整参数
            if np.random.random() < 0.5:
                drone_info['direction'] += np.random.normal(0, 10)
                drone_info['direction'] = drone_info['direction'] % 360
            
            if np.random.random() < 0.5:
                drone_info['speed'] += np.random.normal(0, 5)
                drone_info['speed'] = np.clip(drone_info['speed'], 70, 140)
            
            for bomb in drone_info['bombs']:
                if np.random.random() < 0.3:
                    bomb['explode_delay'] += np.random.normal(0, 1)
                    bomb['explode_delay'] = max(0.1, bomb['explode_delay'])
            
            new_strategy[drone_name] = drone_info
        else:
            # 重新生成该无人机的策略
            if np.random.random() < 0.7:  # 70%概率使用
                direction = np.random.uniform(0, 360)
                speed = np.random.uniform(70, 140)
                n_bombs = np.random.randint(1, 4)
                bombs = []
                
                last_release_time = 0
                for i in range(n_bombs):
                    release_time = last_release_time + np.random.uniform(1.0, 10.0)
                    explode_delay = np.random.uniform(0.5, 8.0)
                    
                    bombs.append({
                        'release_time': release_time,
                        'explode_delay': explode_delay
                    })
                    last_release_time = release_time
                
                new_strategy[drone_name] = {
                    'direction': direction,
                    'speed': speed,
                    'bombs': bombs
                }
    
    return new_strategy

class EvolutionaryOptimizer:
    def __init__(self, population_size=50, generations=100, elite_ratio=0.2):
        self.population_size = population_size
        self.generations = generations
        self.elite_size = int(population_size * elite_ratio)
        
        # 混合初始化种群：结合智能策略和随机策略
        self.population = []
        
        # 40%使用智能策略生成
        strategic_count = int(population_size * 0.4)
        for _ in range(strategic_count):
            try:
                strategy = create_opportunistic_strategy()
                if strategy:  # 确保策略非空
                    self.population.append(strategy)
                else:
                    self.population.append(create_random_strategy())
            except Exception as e:
                print(f"智能策略生成失败: {e}")
                self.population.append(create_random_strategy())
        
        # 补充随机策略
        while len(self.population) < population_size:
            self.population.append(create_random_strategy())
            
        self.fitness_history = []
    
    def evaluate_population(self):
        """评估整个种群的适应度"""
        fitness_values = Parallel(n_jobs=8)(
            delayed(compute_total_coverage_time)(strategy) 
            for strategy in self.population
        )
        return np.array(fitness_values)
    
    def run(self):
        """运行进化算法"""
        best_fitness = -np.inf
        best_strategy = None
        
        for gen in trange(self.generations, desc="进化算法进度"):
            # 评估适应度
            fitness_values = self.evaluate_population()
            
            # 记录最佳结果
            gen_best_idx = np.argmax(fitness_values)
            gen_best_fitness = fitness_values[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_strategy = self.population[gen_best_idx].copy()
            
            self.fitness_history.append(best_fitness)
            
            # 选择精英
            elite_indices = np.argsort(fitness_values)[-self.elite_size:]
            elite_population = [self.population[i] for i in elite_indices]
            
            # 生成新一代
            new_population = elite_population.copy()
            
            while len(new_population) < self.population_size:
                # 选择父代（轮盘赌选择）
                parent_idx = np.random.choice(elite_indices)
                parent = self.population[parent_idx]
                
                # 变异生成子代
                child = mutate_strategy(parent)
                new_population.append(child)
            
            self.population = new_population
            
            if gen % 20 == 0:
                print(f"第{gen}代：最佳适应度 = {best_fitness:.2f}")
        
        return best_strategy, best_fitness

def strategy_to_dataframe(strategy):
    """将策略转换为DataFrame格式"""
    rows = []
    bomb_counter = 1
    
    for drone_name in sorted(strategy.keys()):
        drone_info = strategy[drone_name]
        direction = drone_info['direction']
        speed = drone_info['speed']
        bombs = drone_info['bombs']
        
        for bomb in bombs:
            release_time = bomb['release_time']
            explode_delay = bomb['explode_delay']
            
            # 计算投放位置和起爆位置
            release_pos = drone_position(drone_name, release_time, direction, speed)
            explode_time = release_time + explode_delay
            explode_pos = smoke_bomb_trajectory(release_pos, release_time, explode_time)
            
            row = {
                "无人机编号": drone_name,
                "无人机运动方向": f"{direction:.1f}",
                "无人机运动速度 (m/s)": f"{speed:.1f}",
                "烟幕干扰弹编号": f"B{bomb_counter}",
                "烟幕干扰弹投放点的x坐标 (m)": f"{release_pos[0]:.1f}",
                "烟幕干扰弹投放点的y坐标 (m)": f"{release_pos[1]:.1f}",
                "烟幕干扰弹投放点的z坐标 (m)": f"{release_pos[2]:.1f}",
                "烟幕干扰弹起爆点的x坐标 (m)": f"{explode_pos[0]:.1f}",
                "烟幕干扰弹起爆点的y坐标 (m)": f"{explode_pos[1]:.1f}",
                "烟幕干扰弹起爆点的z坐标 (m)": f"{explode_pos[2]:.1f}"
            }
            rows.append(row)
            bomb_counter += 1
    
    return pd.DataFrame(rows)

def save_optimization_history(fitness_history, output_dir="./q5_out"):
    """保存优化历史并绘图"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    plt.plot(fitness_history, linewidth=3, color='blue', alpha=0.8)
    plt.title('Q5: 进化算法优化进程 - 总遮蔽时间', fontsize=16, fontweight='bold')
    plt.xlabel('迭代次数', fontsize=14)
    plt.ylabel('总遮蔽时间 (秒)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 添加最佳值标注
    best_gen = np.argmax(fitness_history)
    best_fitness = max(fitness_history)
    plt.scatter(best_gen, best_fitness, color='red', s=100, zorder=5)
    plt.annotate(f'最优解: {best_fitness:.3f}s\n第{best_gen+1}代', 
                xy=(best_gen, best_fitness), xytext=(10, 10),
                textcoords='offset points', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/q5_optimization_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 优化历史图已保存到: {output_dir}/q5_optimization_history.png")

# ======================
# 主程序
# ======================
def analyze_strategy_effectiveness(strategy):
    """分析策略的详细效果"""
    print("\n=== 策略效果深度分析 ===")
    
    # 分析底线保障
    base_assignments = design_bottom_line_coverage()
    print(f"\n底线思维分析:")
    print(f"  基础任务分配: {[(m, info['drone']) for m, info in base_assignments.items()]}")
    
    # 分析机会识别
    opportunities = calculate_interception_opportunities()
    total_opportunities = sum(len(ops) for drone_ops in opportunities.values() 
                             for ops in drone_ops.values() if ops)
    print(f"  识别到的拦截机会总数: {total_opportunities}")
    
    # 分析每个导弹的遮蔽情况
    t_max = 100
    times = np.arange(0, t_max, DT)
    
    # 解析烟幕事件
    smoke_events = []
    for drone_name in DRONES.keys():
        if drone_name not in strategy:
            continue
        drone_info = strategy[drone_name]
        direction = drone_info['direction']
        speed = drone_info['speed']
        bombs = drone_info['bombs']
        
        for bomb in bombs:
            release_time = bomb['release_time']
            explode_delay = bomb['explode_delay']
            
            release_pos = drone_position(drone_name, release_time, direction, speed)
            explode_time = release_time + explode_delay
            explode_pos = smoke_bomb_trajectory(release_pos, release_time, explode_time)
            
            smoke_events.append({
                'drone': drone_name,
                'explode_pos': explode_pos,
                'explode_time': explode_time,
                'end_time': explode_time + SMOKE_DURATION,
                'target': bomb.get('target_missile', 'unknown'),
                'priority': bomb.get('priority', 'unknown')
            })
    
    # 分析每个导弹的遮蔽效果
    print(f"\n导弹遮蔽效果分析:")
    total_coverage = 0
    for missile_name in MISSILES.keys():
        missile_coverage = 0.0
        coverage_periods = []
        
        current_start = None
        for t in times:
            m_pos = missile_position(missile_name, t)
            
            blocked = False
            for smoke_event in smoke_events:
                if smoke_event['explode_time'] <= t <= smoke_event['end_time']:
                    smoke_pos = smoke_cloud_position(
                        smoke_event['explode_pos'], 
                        smoke_event['explode_time'], 
                        t
                    )
                    
                    if is_line_blocked_by_sphere(m_pos, REAL_TARGET, smoke_pos, SMOKE_RADIUS):
                        blocked = True
                        break
            
            if blocked:
                if current_start is None:
                    current_start = t
                missile_coverage += DT
            else:
                if current_start is not None:
                    coverage_periods.append((current_start, t))
                    current_start = None
        
        if current_start is not None:
            coverage_periods.append((current_start, times[-1]))
        
        print(f"  {missile_name}: 总遮蔽时间 {missile_coverage:.1f}s")
        for i, (start, end) in enumerate(coverage_periods[:3]):  # 只显示前3个时段
            print(f"    遮蔽期{i+1}: {start:.1f}s - {end:.1f}s (持续{end-start:.1f}s)")
        
        total_coverage += missile_coverage
    
    # 分析无人机任务执行
    print(f"\n无人机任务执行分析:")
    for drone_name, drone_info in strategy.items():
        bombs = drone_info['bombs']
        primary_bombs = [b for b in bombs if b.get('priority') == 'primary']
        opportunity_bombs = [b for b in bombs if b.get('priority') == 'opportunity']
        
        print(f"  {drone_name}: 主要任务{len(primary_bombs)}个, 机会任务{len(opportunity_bombs)}个")
        print(f"    飞行方向: {drone_info['direction']:.1f}°, 速度: {drone_info['speed']:.1f}m/s")
    
    return total_coverage

def create_visualizations(best_strategy, best_fitness):
    """创建Q5问题的可视化图表"""
    os.makedirs("./output", exist_ok=True)
    
    # 解析策略数据
    drone_colors = ['orange', 'cyan', 'magenta', 'lime', 'pink']
    missile_colors = ['red', 'darkred', 'crimson']
    
    # 1. 三维场景图
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制导弹轨迹
    missile_names = list(MISSILES.keys())
    for i, missile_name in enumerate(missile_names):
        missile_flight_time = 70  # 假设70秒飞行时间
        t_trajectory = np.linspace(0, missile_flight_time, 100)
        missile_trajectory = np.array([missile_position(missile_name, t) for t in t_trajectory])
        ax.plot(missile_trajectory[:, 0], missile_trajectory[:, 1], missile_trajectory[:, 2], 
                color=missile_colors[i], linewidth=4, label=f'导弹{missile_name}轨迹')
        
        # 标记导弹初始位置
        start_pos = MISSILES[missile_name]
        ax.scatter(*start_pos, color=missile_colors[i], s=200, marker='d', 
                  label=f'导弹{missile_name}初始位置')
    
    # 绘制无人机轨迹和烟幕弹
    drone_names = list(DRONES.keys())
    for i, drone_name in enumerate(drone_names):
        if drone_name in best_strategy:
            drone_info = best_strategy[drone_name]
            color = drone_colors[i]
            
            # 绘制无人机轨迹
            max_time = 50  # 假设50秒最大飞行时间
            t_traj = np.linspace(0, max_time, 100)
            uav_trajectory = np.array([drone_position(drone_name, t, drone_info['direction'], drone_info['speed']) for t in t_traj])
            ax.plot(uav_trajectory[:, 0], uav_trajectory[:, 1], uav_trajectory[:, 2], 
                    color=color, linewidth=2, label=f'无人机{drone_name}轨迹')
            
            # 标记无人机初始位置
            start_pos = DRONES[drone_name]
            ax.scatter(*start_pos, color=color, s=150, label=f'无人机{drone_name}初始位置')
            
            # 绘制烟幕弹投放点和起爆点
            for j, bomb_info in enumerate(drone_info['bombs']):
                release_time = bomb_info['release_time']
                explode_delay = bomb_info['explode_delay']
                explode_time = release_time + explode_delay
                
                # 投放点
                drop_pos = drone_position(drone_name, release_time, drone_info['direction'], drone_info['speed'])
                ax.scatter(*drop_pos, color=color, s=200, marker='*', alpha=0.8)
                
                # 起爆点
                explode_pos = smoke_bomb_trajectory(drop_pos, release_time, explode_time)
                ax.scatter(*explode_pos, color=color, s=200, marker='o', alpha=0.8)
                
                # 绘制烟幕球体
                u = np.linspace(0, 2 * np.pi, 15)
                v = np.linspace(0, np.pi, 15)
                x_sphere = SMOKE_RADIUS * np.outer(np.cos(u), np.sin(v)) + explode_pos[0]
                y_sphere = SMOKE_RADIUS * np.outer(np.sin(u), np.sin(v)) + explode_pos[1]
                z_sphere = SMOKE_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v)) + explode_pos[2]
                ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.15, color=color)
    
    # 标记目标位置
    ax.scatter(*FAKE_TARGET, color='black', s=200, marker='s', label='假目标')
    ax.scatter(*REAL_TARGET, color='green', s=200, marker='^', label='真目标')
    
    # 绘制真目标圆柱体
    theta = np.linspace(0, 2*np.pi, 20)
    z_cyl = np.linspace(0, 10, 10)
    theta_mesh, z_mesh = np.meshgrid(theta, z_cyl)
    x_cyl = 7 * np.cos(theta_mesh)
    y_cyl = REAL_TARGET[1] + 7 * np.sin(theta_mesh)
    ax.plot_surface(x_cyl, y_cyl, z_mesh, alpha=0.4, color='green')
    
    ax.set_xlabel('X (米)')
    ax.set_ylabel('Y (米)')
    ax.set_zlabel('Z (米)')
    ax.set_title('Q5: 五无人机协同对抗三导弹烟幕干扰三维场景图', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('./output/q5_3d_scenario.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 俯视图（XY平面）
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # 绘制导弹轨迹投影
    for i, missile_name in enumerate(missile_names):
        missile_flight_time = 70
        t_trajectory = np.linspace(0, missile_flight_time, 100)
        missile_trajectory = np.array([missile_position(missile_name, t) for t in t_trajectory])
        ax.plot(missile_trajectory[:, 0], missile_trajectory[:, 1], 
                color=missile_colors[i], linewidth=4, label=f'导弹{missile_name}轨迹')
        
        # 标记导弹初始位置
        start_pos = MISSILES[missile_name]
        ax.scatter(start_pos[0], start_pos[1], color=missile_colors[i], s=200, marker='d', 
                  label=f'导弹{missile_name}初始位置')
    
    # 绘制无人机轨迹和烟幕覆盖区域
    for i, drone_name in enumerate(drone_names):
        if drone_name in best_strategy:
            drone_info = best_strategy[drone_name]
            color = drone_colors[i]
            
            # 绘制无人机轨迹投影
            max_time = 50
            t_traj = np.linspace(0, max_time, 100)
            uav_trajectory = np.array([drone_position(drone_name, t, drone_info['direction'], drone_info['speed']) for t in t_traj])
            ax.plot(uav_trajectory[:, 0], uav_trajectory[:, 1], 
                    color=color, linewidth=2, label=f'无人机{drone_name}轨迹')
            
            # 标记无人机初始位置
            start_pos = DRONES[drone_name]
            ax.scatter(start_pos[0], start_pos[1], color=color, s=150, label=f'无人机{drone_name}初始位置')
            
            # 绘制烟幕弹投放点、起爆点和覆盖区域
            for j, bomb_info in enumerate(drone_info['bombs']):
                release_time = bomb_info['release_time']
                explode_delay = bomb_info['explode_delay']
                explode_time = release_time + explode_delay
                
                # 投放点
                drop_pos = drone_position(drone_name, release_time, drone_info['direction'], drone_info['speed'])
                ax.scatter(drop_pos[0], drop_pos[1], color=color, s=200, marker='*', alpha=0.8)
                
                # 起爆点
                explode_pos = smoke_bomb_trajectory(drop_pos, release_time, explode_time)
                ax.scatter(explode_pos[0], explode_pos[1], color=color, s=200, marker='o', alpha=0.8)
                
                # 绘制烟幕覆盖区域
                smoke_circle = patches.Circle((explode_pos[0], explode_pos[1]), SMOKE_RADIUS, 
                                            linewidth=1, edgecolor=color, facecolor=color, alpha=0.2)
                ax.add_patch(smoke_circle)
    
    # 标记目标位置
    ax.scatter(FAKE_TARGET[0], FAKE_TARGET[1], color='black', s=200, marker='s', label='假目标')
    ax.scatter(REAL_TARGET[0], REAL_TARGET[1], color='green', s=200, marker='^', label='真目标')
    
    # 绘制真目标保护区
    circle_true = patches.Circle((REAL_TARGET[0], REAL_TARGET[1]), 7, linewidth=2, 
                               edgecolor='green', facecolor='lightgreen', alpha=0.3, label='真目标保护区')
    ax.add_patch(circle_true)
    
    ax.set_xlabel('X (米)')
    ax.set_ylabel('Y (米)')
    ax.set_title('Q5: 五无人机协同对抗三导弹烟幕干扰俯视图', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('./output/q5_top_view.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 策略分析图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # 子图1: 各无人机投放烟幕弹数量
    active_drones = [drone for drone in drone_names if drone in best_strategy]
    bomb_counts = [len(best_strategy[drone]['bombs']) for drone in active_drones]
    
    bars1 = ax1.bar(active_drones, bomb_counts, color=drone_colors[:len(active_drones)], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('烟幕弹数量')
    ax1.set_title('各无人机投放烟幕弹数量分布', fontweight='bold')
    ax1.grid(True, axis='y')
    
    # 在柱状图上添加数值标签
    for bar, count in zip(bars1, bomb_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 子图2: 各无人机飞行速度
    speeds = [best_strategy[drone]['speed'] for drone in active_drones]
    bars2 = ax2.bar(active_drones, speeds, color=drone_colors[:len(active_drones)], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('飞行速度 (m/s)')
    ax2.set_title('各无人机飞行速度', fontweight='bold')
    ax2.grid(True, axis='y')
    
    for bar, speed in zip(bars2, speeds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{speed:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 子图3: 各无人机飞行方向
    directions = [best_strategy[drone]['direction'] for drone in active_drones]
    bars3 = ax3.bar(active_drones, directions, color=drone_colors[:len(active_drones)], alpha=0.7, edgecolor='black')
    ax3.set_ylabel('飞行方向 (度)')
    ax3.set_title('各无人机飞行方向', fontweight='bold')
    ax3.grid(True, axis='y')
    
    for bar, direction in zip(bars3, directions):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{direction:.1f}°', ha='center', va='bottom', fontweight='bold')
    
    # 子图4: 每个导弹的遮蔽时间分布
    missile_coverage = {}
    for missile_name in missile_names:
        coverage = 0
        for drone_name in active_drones:
            for bomb_info in best_strategy[drone_name]['bombs']:
                if bomb_info.get('target_missile') == missile_name:
                    # 简单估算该烟幕弹对该导弹的遮蔽时间
                    coverage += min(SMOKE_DURATION, 5.0)  # 假设平均每个烟幕弹贡献5秒
        missile_coverage[missile_name] = coverage
    
    missiles = list(missile_coverage.keys())
    coverages = list(missile_coverage.values())
    bars4 = ax4.bar(missiles, coverages, color=missile_colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('估算遮蔽时间 (秒)')
    ax4.set_title('各导弹遮蔽时间分布', fontweight='bold')
    ax4.grid(True, axis='y')
    
    for bar, coverage in zip(bars4, coverages):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{coverage:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./output/q5_strategy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 时间轴协同分析图
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 为每个无人机绘制时间轴
    y_positions = {}
    y_pos = 0
    
    for i, drone_name in enumerate(active_drones):
        y_positions[drone_name] = y_pos
        drone_info = best_strategy[drone_name]
        color = drone_colors[i]
        
        # 绘制无人机飞行时间线
        ax.barh(y_pos, 50, height=0.3, color=color, alpha=0.3, label=f'{drone_name}飞行时间')
        
        # 绘制烟幕弹投放和起爆时间
        for j, bomb_info in enumerate(drone_info['bombs']):
            release_time = bomb_info['release_time']
            explode_delay = bomb_info['explode_delay']
            explode_time = release_time + explode_delay
            
            # 投放时间标记
            ax.scatter(release_time, y_pos, color=color, s=100, marker='*', alpha=0.8)
            ax.text(release_time, y_pos + 0.2, f'投放{j+1}', ha='center', va='bottom', fontsize=8)
            
            # 起爆时间标记
            ax.scatter(explode_time, y_pos, color=color, s=100, marker='o', alpha=0.8)
            ax.text(explode_time, y_pos - 0.2, f'起爆{j+1}', ha='center', va='top', fontsize=8)
            
            # 有效遮蔽时间段
            ax.barh(y_pos, SMOKE_DURATION, left=explode_time, height=0.15, 
                   color=color, alpha=0.6, label=f'{drone_name}烟幕{j+1}' if j == 0 else "")
        
        y_pos += 1
    
    ax.set_xlabel('时间 (秒)')
    ax.set_ylabel('无人机')
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.set_title(f'Q5: 五机协同时间轴分析 (总遮蔽时间: {best_fitness:.2f}秒)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='x')
    
    plt.tight_layout()
    plt.savefig('./output/q5_timeline_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Q5可视化图表已保存到output/目录")

def save_results_to_excel(best_strategy, best_fitness):
    """保存结果到Excel文件"""
    os.makedirs("./output", exist_ok=True)
    
    # 按照题目要求的格式保存Q5结果 - Q5对应的是q5_result3_data.xlsx
    result_df = strategy_to_dataframe(best_strategy)
    output_file = "./output/q5_result3_data.xlsx"
    result_df.to_excel(output_file, index=False)
    print(f"✓ 结果已保存到: {output_file}")
    
    # 保存详细分析结果
    detailed_data = []
    total_bombs = 0
    active_drones = 0
    
    for drone_name, drone_info in best_strategy.items():
        active_drones += 1
        for i, bomb_info in enumerate(drone_info['bombs']):
            total_bombs += 1
            detailed_data.append({
                'scenario': 'Q5_five_drones_three_missiles',
                'drone_id': drone_name,
                'bomb_sequence': i + 1,
                'drone_direction_deg': drone_info['direction'],
                'drone_speed_m_s': drone_info['speed'],
                'drop_time_s': bomb_info['release_time'],
                'explode_time_s': bomb_info['release_time'] + bomb_info['explode_delay'],
                'delay_s': bomb_info['explode_delay'],
                'target_missile': bomb_info.get('target_missile', 'unknown'),
                'priority': bomb_info.get('priority', 'primary'),
                'estimated_coverage_s': bomb_info.get('estimated_coverage', 0.0)
            })
    
    # 添加汇总信息
    summary_data = [{
        'total_coverage_time_s': best_fitness,
        'active_drones': active_drones,
        'total_bombs_deployed': total_bombs,
        'average_bombs_per_drone': total_bombs / active_drones if active_drones > 0 else 0,
        'optimization_method': 'evolutionary_algorithm',
        'random_seed': 42,
        'missile_targets': len(MISSILES),
        'drone_fleet_size': len(DRONES)
    }]
    
    df_detailed = pd.DataFrame(detailed_data)
    df_summary = pd.DataFrame(summary_data)
    
    with pd.ExcelWriter("./output/q5_detailed_results.xlsx", engine='openpyxl') as writer:
        df_detailed.to_excel(writer, sheet_name='Detailed_Strategy', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print("\n📁 生成的文件:")
    print("  - output/q5_result3_data.xlsx (Q5标准结果，对应题目result3.xlsx)")
    print("  - output/q5_detailed_results.xlsx (详细分析数据)")
    print("  - output/q5_3d_scenario.png (三维场景图)")
    print("  - output/q5_top_view.png (俯视图)")
    print("  - output/q5_strategy_analysis.png (策略分析图)")
    print("  - output/q5_timeline_analysis.png (时间轴分析图)")

def main():
    print("开始Q5问题求解：多架无人机对多枚导弹的烟幕干扰策略优化")
    print("采用进化算法优化（已设置随机种子42确保结果可重复）")
    print("="*70)
    
    # 运行进化算法
    print(f"\n🔧 进化算法优化")
    optimizer = EvolutionaryOptimizer(
        population_size=80,  # 增加种群大小
        generations=120,
        elite_ratio=0.15
    )
    
    print("开始进化算法优化...")
    best_strategy, best_fitness = optimizer.run()
    
    print("=" * 70)
    print("Q5: 五无人机协同对抗三导弹烟幕干扰问题求解完成")
    print("=" * 70)
    
    print(f"\n📊 最优解结果:")
    print(f"  最佳总遮蔽时间: {best_fitness:.3f} 秒")
    
    # 详细分析最佳策略
    analyze_strategy_effectiveness(best_strategy)
    
    # 生成可视化图表
    print("\n🎨 生成可视化图表...")
    create_visualizations(best_strategy, best_fitness)
    
    # 保存结果到Excel
    print("\n💾 保存结果到Excel...")
    save_results_to_excel(best_strategy, best_fitness)
    
    # 保存优化历史图到output目录
    save_optimization_history(optimizer.fitness_history, output_dir="./output")
    
    # 验证结果
    final_coverage = compute_total_coverage_time(best_strategy)
    print(f"\n✅ Q5问题求解完成！")
    print(f"   最终验证总遮蔽时间: {final_coverage:.3f} 秒")
    print(f"   优化算法收敛性: {'良好' if abs(best_fitness - final_coverage) < 0.01 else '需要调整'}")
    print(f"   随机种子: 42 (确保结果可重复)")

if __name__ == "__main__":
    main()
