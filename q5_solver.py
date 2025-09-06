import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import trange
import os
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

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

def save_optimization_history(fitness_history):
    """保存优化历史并绘图"""
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, linewidth=2)
    plt.title('优化进程 - 总遮蔽时间', fontsize=14)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('总遮蔽时间 (秒)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./q5_out/q5_optimization_history.png', dpi=300, bbox_inches='tight')
    plt.close()

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

def main():
    print("开始Q5问题求解：多架无人机对多枚导弹的烟幕干扰策略优化")
    print("采用'化曲为直'和'底线思维'的智能策略")
    print("="*70)
    
    os.makedirs("./q5_out", exist_ok=True)
    
    # 先展示策略分析
    print("\n第一步：智能策略分析")
    try:
        sample_strategy = create_opportunistic_strategy()
        if sample_strategy:
            sample_fitness = compute_total_coverage_time(sample_strategy)
            print(f"智能策略初始效果: {sample_fitness:.2f} 秒")
        else:
            print("智能策略生成为空，将使用纯随机优化")
    except Exception as e:
        print(f"智能策略分析失败: {e}")
    
    # 运行进化算法
    print(f"\n第二步：进化算法优化")
    optimizer = EvolutionaryOptimizer(
        population_size=80,  # 增加种群大小以利用智能策略
        generations=120,
        elite_ratio=0.15
    )
    
    print("开始进化算法优化...")
    best_strategy, best_fitness = optimizer.run()
    
    print(f"\n优化完成！")
    print(f"最佳总遮蔽时间: {best_fitness:.2f} 秒")
    
    # 详细分析最佳策略
    analyze_strategy_effectiveness(best_strategy)
    
    # 保存结果
    result_df = strategy_to_dataframe(best_strategy)
    result_df.to_excel("./q5_out/result3.xlsx", index=False)
    
    # 保存优化历史图
    save_optimization_history(optimizer.fitness_history)
    
    print(f"\n结果已保存到:")
    print(f"  - ./q5_out/result3.xlsx (主要结果)")
    print(f"  - ./q5_out/q5_optimization_history.png (优化过程图)")
    
    # 验证结果
    final_coverage = compute_total_coverage_time(best_strategy)
    print(f"\n最终验证：重新计算的总遮蔽时间 = {final_coverage:.2f} 秒")

if __name__ == "__main__":
    main()
