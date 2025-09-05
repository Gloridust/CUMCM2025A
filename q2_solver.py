#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
烟幕干扰弹投放策略优化器 - 问题2求解器
Problem 2: 单架无人机最优烟幕干扰策略

基于事件驱动边界检测和坐标下降优化的高效求解器
采用严格几何建模和高精度数值计算方法

作者: CUMCM2025团队
版本: 2.0 (优化版)
"""

import numpy as np
from math import cos, sin, pi, sqrt
import time
from tqdm import tqdm
import sys
import os

# 设置随机种子确保可重现性
RNG = np.random.default_rng(2025)

# ==================== 物理常量和场景参数 ====================
class PhysicsConstants:
    """物理常量定义"""
    GRAVITY = 9.8                    # 重力加速度 (m/s²)
    SMOKE_RADIUS = 10.0             # 烟幕球半径 (m)
    CLOUD_SINK_SPEED = 3.0          # 云团中心下沉速度 (m/s)
    CLOUD_DURATION = 20.0           # 起爆后有效持续时间 (s)
    FUSE_DELAY = 3.6                # 投放到起爆延时 (s)
    
class ScenarioConfig:
    """场景配置参数"""
    # 速度约束
    UAV_SPEED_MIN = 70.0            # UAV最小速度 (m/s)
    UAV_SPEED_MAX = 140.0           # UAV最大速度 (m/s)
    
    # 目标和初始位置
    TRUE_TARGET = np.array([0.0, 200.0, 0.0])       # 真目标位置
    MISSILE_START = np.array([20000.0, 0.0, 2000.0]) # 导弹初始位置
    UAV_START = np.array([17800.0, 0.0, 1800.0])     # UAV初始位置
    MISSILE_SPEED = 300.0                            # 导弹速度 (m/s)
    
    # 起爆时刻范围（从任务开始t=0起）
    EXPLOSION_TIME_MIN = PhysicsConstants.FUSE_DELAY  # 最早起爆时刻
    EXPLOSION_TIME_MAX = 20.0                         # 最晚起爆时刻

class OptimizationConfig:
    """优化算法配置"""
    # 高精度模式配置
    RANDOM_SEEDS_COUNT = 500        # 随机种子数量 (提升至500)
    TOP_SEEDS_REFINE = 5           # 精化的顶级种子数量 (提升至5)
    MAX_ITERATIONS = 120           # 坐标下降最大迭代次数 (提升至120)
    
    # 高精度数值参数
    COARSE_SCAN_DT = 0.04          # 粗扫描时间步长 (减小至0.04)
    FINE_SCAN_DT = 0.015           # 精细扫描时间步长 (减小至0.015)
    BOUNDARY_TOLERANCE = 5e-5      # 边界求解容忍度 (提升精度)
    FINE_TOLERANCE = 2e-5          # 精细评估容忍度 (提升精度)
    
    # 超精细模式参数
    ULTRA_FINE_SCAN_DT = 0.008     # 超精细扫描时间步长
    ULTRA_FINE_TOLERANCE = 8e-6    # 超精细评估容忍度
    
    @classmethod
    def get_precision_mode(cls, mode: str = "high"):
        """获取不同精度模式的参数配置
        
        Args:
            mode: 精度模式 ('standard', 'high', 'ultra')
        
        Returns:
            dict: 配置参数字典
        """
        if mode == "standard":
            return {
                'seeds_count': 200,
                'top_refine': 3,
                'max_iter': 80,
                'coarse_dt': 0.08,
                'fine_dt': 0.03,
                'boundary_tol': 1e-4,
                'fine_tol': 6e-5
            }
        elif mode == "high":
            return {
                'seeds_count': 500,
                'top_refine': 5,
                'max_iter': 120,
                'coarse_dt': 0.04,
                'fine_dt': 0.015,
                'boundary_tol': 5e-5,
                'fine_tol': 2e-5
            }
        elif mode == "ultra":
            return {
                'seeds_count': 800,
                'top_refine': 8,
                'max_iter': 150,
                'coarse_dt': 0.02,
                'fine_dt': 0.008,
                'boundary_tol': 2e-5,
                'fine_tol': 8e-6
            }
        else:
            raise ValueError(f"未知精度模式: {mode}")

# ==================== 全局变量 ====================
g_function_calls = 0  # 目标函数调用计数器

# ==================== 工具函数 ====================
def format_number(x: float, decimals: int = 3) -> float:
    """格式化数字到指定小数位数"""
    return float(f"{float(x):.{decimals}f}")

def format_vector(vector) -> tuple:
    """格式化向量到3位小数"""
    return tuple(format_number(x) for x in vector)

def wrap_angle_to_2pi(angle: float) -> float:
    """将角度规范化到[0, 2π)范围"""
    return (angle + 2*pi) % (2*pi)

def clamp_speed_by_time(distance: float, explosion_time: float) -> float:
    """根据起爆时间约束速度对应的距离"""
    min_distance = ScenarioConfig.UAV_SPEED_MIN * explosion_time
    max_distance = ScenarioConfig.UAV_SPEED_MAX * explosion_time
    return np.clip(distance, min_distance, max_distance)

# ==================== 几何计算模块 ====================
class GeometryCalculator:
    """几何计算相关函数集合"""
    
    @staticmethod
    def get_missile_unit_direction():
        """计算导弹朝向假目标的单位方向向量"""
        direction = -ScenarioConfig.MISSILE_START
        return direction / np.linalg.norm(direction)
    
    @staticmethod
    def get_missile_position(time: float):
        """计算导弹在时刻t的位置"""
        unit_dir = GeometryCalculator.get_missile_unit_direction()
        return ScenarioConfig.MISSILE_START + ScenarioConfig.MISSILE_SPEED * unit_dir * time
    
    @staticmethod
    def point_to_segment_distance(point, segment_start, segment_end):
        """计算点到线段的最短距离"""
        segment_vector = segment_end - segment_start
        denominator = float(np.dot(segment_vector, segment_vector))
        
        if denominator == 0.0:
            return float(np.linalg.norm(point - segment_start))
        
        t = float(np.dot(point - segment_start, segment_vector) / denominator)
        t = np.clip(t, 0.0, 1.0)
        
        closest_point = segment_start + t * segment_vector
        return float(np.linalg.norm(point - closest_point))
    
    @staticmethod
    def explosion_position_from_polar(explosion_time: float, angle: float, distance: float):
        """从极坐标参数计算起爆位置"""
        x_explosion = ScenarioConfig.UAV_START[0] + distance * cos(angle)
        y_explosion = ScenarioConfig.UAV_START[1] + distance * sin(angle)
        return x_explosion, y_explosion

# ==================== 物理模拟模块 ====================
class PhysicsSimulator:
    """物理模拟相关函数"""
    
    @staticmethod
    def get_cloud_center_position(time: float, explosion_x: float, explosion_y: float, 
                                explosion_z: float, explosion_time: float):
        """计算云团中心在时刻t的位置"""
        if time < explosion_time:
            return None
        
        sink_distance = PhysicsConstants.CLOUD_SINK_SPEED * (time - explosion_time)
        return np.array([explosion_x, explosion_y, explosion_z - sink_distance], dtype=float)
    
    @staticmethod
    def calculate_occlusion_function(time: float, explosion_x: float, explosion_y: float, 
                                   explosion_z: float, explosion_time: float):
        """
        计算遮蔽函数g(t)的值
        返回值 <= 0 表示存在遮蔽，> 0 表示无遮蔽
        """
        global g_function_calls
        g_function_calls += 1
        
        cloud_center = PhysicsSimulator.get_cloud_center_position(
            time, explosion_x, explosion_y, explosion_z, explosion_time)
        
        if cloud_center is None:
            return float('inf')  # 起爆前无遮蔽
        
        missile_position = GeometryCalculator.get_missile_position(time)
        distance_to_line = GeometryCalculator.point_to_segment_distance(
            cloud_center, missile_position, ScenarioConfig.TRUE_TARGET)
        
        return distance_to_line - PhysicsConstants.SMOKE_RADIUS

# ==================== 数值算法模块 ====================
class NumericalSolver:
    """数值求解算法集合"""
    
    @staticmethod
    def bisection_root_finding(func, left_bound: float, right_bound: float, 
                             func_left: float, func_right: float, 
                             tolerance: float = 1e-4, max_iterations: int = 60):
        """二分法求根"""
        if func_left == 0.0:
            return left_bound
        if func_right == 0.0:
            return right_bound
        
        low, high = left_bound, right_bound
        f_low, f_high = func_left, func_right
        
        for _ in range(max_iterations):
            mid = 0.5 * (low + high)
            f_mid = func(mid)
            
            if abs(f_mid) < 1e-12 or (high - low) < tolerance:
                return mid
                
            if f_low * f_mid <= 0.0:
                high, f_high = mid, f_mid
            else:
                low, f_low = mid, f_mid
                
        return 0.5 * (low + high)
    
    @staticmethod
    def find_occlusion_intervals(explosion_time: float, explosion_x: float, explosion_y: float,
                               scan_step: float = 0.08, tolerance: float = 1e-4):
        """
        事件驱动方法寻找遮蔽时间区间
        返回: (总遮蔽时间, 区间列表)
        """
        # 计算起爆点的z坐标（考虑自由落体）
        explosion_z = ScenarioConfig.UAV_START[2] - 0.5 * PhysicsConstants.GRAVITY * (PhysicsConstants.FUSE_DELAY ** 2)
        
        # 分析时间窗口
        time_start = explosion_time
        time_end = explosion_time + PhysicsConstants.CLOUD_DURATION
        
        if time_end <= time_start:
            return 0.0, []
        
        # 创建遮蔽函数
        def occlusion_func(t):
            return PhysicsSimulator.calculate_occlusion_function(
                t, explosion_x, explosion_y, explosion_z, explosion_time)
        
        # 粗扫描寻找符号变化点
        time_points = [time_start]
        current_time = time_start
        while current_time < time_end:
            current_time = min(current_time + scan_step, time_end)
            time_points.append(current_time)
        
        function_values = [occlusion_func(t) for t in time_points]
        
        # 早期退出优化：如果距离过远则无遮蔽
        if min(function_values) > 8.0:
            return 0.0, []
        
        # 使用二分法精确定位边界
        boundary_times = []
        for i in range(1, len(time_points)):
            t_prev, t_curr = time_points[i-1], time_points[i]
            f_prev, f_curr = function_values[i-1], function_values[i]
            
            if f_prev == 0.0:
                boundary_times.append(t_prev)
            if f_prev * f_curr < 0.0:
                root = NumericalSolver.bisection_root_finding(
                    occlusion_func, t_prev, t_curr, f_prev, f_curr, tolerance)
                boundary_times.append(root)
        
        boundary_times = sorted(set(boundary_times))
        
        # 构建遮蔽区间（g <= 0的区间）
        intervals = []
        is_occluded = (function_values[0] <= 0.0)
        interval_start = time_start if is_occluded else None
        
        for boundary_time in boundary_times:
            if is_occluded:
                intervals.append((interval_start, boundary_time))
                is_occluded, interval_start = False, None
            else:
                is_occluded, interval_start = True, boundary_time
        
        if is_occluded:
            intervals.append((interval_start, time_end))
        
        total_occlusion_time = sum(end - start for start, end in intervals)
        return total_occlusion_time, intervals

# ==================== 优化算法模块 ====================
class OptimizationEngine:
    """优化算法引擎"""
    
    @staticmethod
    def objective_function(explosion_time: float, angle: float, distance: float,
                         scan_step: float = 0.08, tolerance: float = 1e-4):
        """目标函数：计算总遮蔽时间"""
        # 参数约束
        explosion_time = float(np.clip(explosion_time, 
                                     ScenarioConfig.EXPLOSION_TIME_MIN, 
                                     ScenarioConfig.EXPLOSION_TIME_MAX))
        angle = wrap_angle_to_2pi(float(angle))
        distance = clamp_speed_by_time(float(distance), explosion_time)
        
        # 计算起爆位置
        explosion_x, explosion_y = GeometryCalculator.explosion_position_from_polar(
            explosion_time, angle, distance)
        
        # 计算遮蔽时间
        total_time, _ = NumericalSolver.find_occlusion_intervals(
            explosion_time, explosion_x, explosion_y, scan_step, tolerance)
        
        return total_time
    
    @staticmethod
    def evaluate_solution_detailed(explosion_time: float, angle: float, distance: float,
                                 scan_step: float = 0.04, tolerance: float = 8e-5):
        """详细评估解的质量，返回完整信息"""
        # 参数约束
        explosion_time = float(np.clip(explosion_time, 
                                     ScenarioConfig.EXPLOSION_TIME_MIN, 
                                     ScenarioConfig.EXPLOSION_TIME_MAX))
        angle = wrap_angle_to_2pi(float(angle))
        distance = clamp_speed_by_time(float(distance), explosion_time)
        
        # 计算起爆位置
        explosion_x, explosion_y = GeometryCalculator.explosion_position_from_polar(
            explosion_time, angle, distance)
        
        # 计算遮蔽时间和区间
        total_time, intervals = NumericalSolver.find_occlusion_intervals(
            explosion_time, explosion_x, explosion_y, scan_step, tolerance)
        
        return total_time, intervals, explosion_time, angle, distance, explosion_x, explosion_y
    
    @staticmethod
    def evaluate_solution_ultra_fine(explosion_time: float, angle: float, distance: float):
        """超精细评估解的质量，用于最终验证"""
        return OptimizationEngine.evaluate_solution_detailed(
            explosion_time, angle, distance,
            scan_step=OptimizationConfig.ULTRA_FINE_SCAN_DT,
            tolerance=OptimizationConfig.ULTRA_FINE_TOLERANCE)
    
    @staticmethod
    def generate_random_seeds(num_seeds: int = 200):
        """生成随机初始解种子"""
        print(f"🌱 正在生成 {num_seeds} 个随机种子解...")
        
        seeds = []
        with tqdm(total=num_seeds, desc="种子生成", ncols=80) as pbar:
            for _ in range(num_seeds):
                explosion_time = RNG.uniform(ScenarioConfig.EXPLOSION_TIME_MIN, 
                                           ScenarioConfig.EXPLOSION_TIME_MAX)
                angle = RNG.uniform(0.0, 2*pi)
                distance = RNG.uniform(ScenarioConfig.UAV_SPEED_MIN * explosion_time,
                                     ScenarioConfig.UAV_SPEED_MAX * explosion_time)
                
                objective_value = OptimizationEngine.objective_function(
                    explosion_time, angle, distance, 
                    scan_step=0.10, tolerance=2e-4)
                
                seeds.append((objective_value, explosion_time, angle, distance))
                pbar.update(1)
        
        # 按目标函数值降序排序
        seeds.sort(key=lambda x: x[0], reverse=True)
        return seeds
    
    @staticmethod
    def coordinate_descent_optimization(initial_explosion_time: float, initial_angle: float, 
                                      initial_distance: float,
                                      time_step_initial: float = 0.3,
                                      angle_step_initial: float = np.deg2rad(1.0),
                                      distance_step_initial: float = 60.0,
                                      time_step_min: float = 0.02,
                                      angle_step_min: float = np.deg2rad(0.05),
                                      distance_step_min: float = 2.0,
                                      max_iterations: int = 120):
        """
        坐标下降优化算法
        对爆炸时间、角度、距离逐个优化，无改善时减小步长
        """
        current_best = OptimizationEngine.objective_function(
            initial_explosion_time, initial_angle, initial_distance,
            scan_step=0.08, tolerance=1e-4)
        
        explosion_time, angle, distance = initial_explosion_time, initial_angle, initial_distance
        time_step, angle_step, distance_step = time_step_initial, angle_step_initial, distance_step_initial
        
        for iteration in range(max_iterations):
            improved = False
            
            # 对每个坐标方向尝试优化
            coordinate_info = [
                (time_step, 'explosion_time'),
                (angle_step, 'angle'), 
                (distance_step, 'distance')
            ]
            
            for step_size, coordinate_name in coordinate_info:
                for direction in [+1, -1]:
                    if coordinate_name == 'explosion_time':
                        candidate_value = OptimizationEngine.objective_function(
                            explosion_time + direction * step_size, angle, distance,
                            scan_step=0.08, tolerance=1e-4)
                        
                        if candidate_value > current_best + 1e-3:
                            current_best = candidate_value
                            explosion_time = np.clip(explosion_time + direction * step_size,
                                                   ScenarioConfig.EXPLOSION_TIME_MIN,
                                                   ScenarioConfig.EXPLOSION_TIME_MAX)
                            distance = clamp_speed_by_time(distance, explosion_time)
                            improved = True
                            break
                            
                    elif coordinate_name == 'angle':
                        candidate_value = OptimizationEngine.objective_function(
                            explosion_time, angle + direction * step_size, distance,
                            scan_step=0.08, tolerance=1e-4)
                        
                        if candidate_value > current_best + 1e-3:
                            current_best = candidate_value
                            angle = wrap_angle_to_2pi(angle + direction * step_size)
                            improved = True
                            break
                            
                    else:  # distance
                        candidate_value = OptimizationEngine.objective_function(
                            explosion_time, angle, distance + direction * step_size,
                            scan_step=0.08, tolerance=1e-4)
                        
                        if candidate_value > current_best + 1e-3:
                            current_best = candidate_value
                            distance = clamp_speed_by_time(distance + direction * step_size, explosion_time)
                            improved = True
                            break
                
                if improved:
                    break
            
            # 如果没有改善，减小步长
            if not improved:
                time_step *= 0.5
                angle_step *= 0.5
                distance_step *= 0.5
                
                # 检查收敛条件
                if (time_step < time_step_min and 
                    angle_step < angle_step_min and 
                    distance_step < distance_step_min):
                    break
        
        return explosion_time, angle, distance, current_best

# ==================== 结果处理模块 ====================
class ResultProcessor:
    """结果处理和输出"""
    
    @staticmethod
    def derive_uav_parameters(explosion_time: float, angle: float, distance: float,
                            explosion_x: float, explosion_y: float):
        """从优化参数反推UAV运动参数"""
        # 计算UAV移动距离和速度
        displacement_x = explosion_x - ScenarioConfig.UAV_START[0]
        displacement_y = explosion_y - ScenarioConfig.UAV_START[1]
        flight_distance = sqrt(displacement_x**2 + displacement_y**2)
        uav_speed = flight_distance / explosion_time
        
        # 计算UAV航向角
        heading_angle = np.arctan2(displacement_y, displacement_x)
        
        # 计算投放时刻
        drop_time = explosion_time - PhysicsConstants.FUSE_DELAY
        
        # UAV单位方向向量
        uav_direction = np.array([cos(heading_angle), sin(heading_angle), 0.0], dtype=float)
        
        # 计算关键位置点
        explosion_z = (ScenarioConfig.UAV_START[2] - 
                      0.5 * PhysicsConstants.GRAVITY * (PhysicsConstants.FUSE_DELAY**2))
        
        drop_position = (ScenarioConfig.UAV_START + 
                        uav_speed * uav_direction * max(0.0, drop_time))
        explosion_position = np.array([explosion_x, explosion_y, explosion_z], dtype=float)
        
        return {
            'uav_speed': uav_speed,
            'heading_angle': heading_angle,
            'uav_direction': uav_direction,
            'drop_time': drop_time,
            'drop_position': drop_position,
            'explosion_position': explosion_position
        }
    
    @staticmethod
    def print_optimization_results(total_time: float, intervals: list, 
                                 explosion_time: float, angle: float, distance: float,
                                 explosion_x: float, explosion_y: float):
        """打印优化结果的详细信息"""
        print("\n" + "="*70)
        print("🎯 问题2最优解 - 单架无人机烟幕干扰策略")
        print("="*70)
        
        # 基本优化结果
        print(f"📊 优化结果:")
        print(f"   • 最大遮蔽时间: {format_number(total_time)} 秒")
        print(f"   • 遮蔽时间区间: {[(format_number(a), format_number(b)) for a, b in intervals]}")
        print()
        
        # 反推UAV参数
        uav_params = ResultProcessor.derive_uav_parameters(
            explosion_time, angle, distance, explosion_x, explosion_y)
        
        print(f"🚁 无人机运动参数:")
        print(f"   • UAV速度: {format_number(uav_params['uav_speed'])} m/s")
        print(f"   • 速度约束范围: [{ScenarioConfig.UAV_SPEED_MIN}, {ScenarioConfig.UAV_SPEED_MAX}] m/s")
        print(f"   • UAV航向角: {format_number(uav_params['heading_angle'])} 弧度")
        print(f"   • UAV方向向量: {format_vector(uav_params['uav_direction'])}")
        print()
        
        print(f"⏱️ 关键时间节点:")
        print(f"   • 投放时刻: {format_number(uav_params['drop_time'])} 秒")
        print(f"   • 起爆时刻: {format_number(explosion_time)} 秒")
        print()
        
        print(f"📍 关键位置坐标:")
        print(f"   • 投放点: {format_vector(uav_params['drop_position'])}")
        print(f"   • 起爆点: {format_vector(uav_params['explosion_position'])}")
        print(f"   • 导弹单位方向: {format_vector(GeometryCalculator.get_missile_unit_direction())}")
        print()

# ==================== 主程序 ====================
def main():
    """主优化流程"""
    print("🚀 烟幕干扰弹投放策略优化器 v2.0")
    print("=" * 70)
    print("📋 问题2: 单架无人机最优烟幕干扰策略求解")
    print("🔬 算法: 事件驱动边界检测 + 坐标下降优化")
    print("=" * 70)
    
    start_time = time.time()
    
    # 第一阶段: 随机种子生成
    print("\n🎲 第一阶段: 随机种子解生成")
    seed_start_time = time.time()
    
    seeds = OptimizationEngine.generate_random_seeds(OptimizationConfig.RANDOM_SEEDS_COUNT)
    
    seed_duration = time.time() - seed_start_time
    print(f"✅ 种子生成完成! 用时: {seed_duration:.2f}秒")
    print(f"🏆 最佳种子解: {format_number(seeds[0][0])} 秒")
    
    # 第二阶段: 局部优化精化
    print(f"\n🔧 第二阶段: 对前 {OptimizationConfig.TOP_SEEDS_REFINE} 个种子进行局部优化")
    
    best_total_time = -1.0
    best_solution_pack = None
    
    for i in range(min(OptimizationConfig.TOP_SEEDS_REFINE, len(seeds))):
        _, initial_explosion_time, initial_angle, initial_distance = seeds[i]
        
        print(f"\n🔍 精化种子 #{i+1}:")
        print(f"   初始解质量: {format_number(seeds[i][0])} 秒")
        
        refine_start_time = time.time()
        
        # 坐标下降优化
        optimized_explosion_time, optimized_angle, optimized_distance, optimized_value = \
            OptimizationEngine.coordinate_descent_optimization(
                initial_explosion_time, initial_angle, initial_distance)
        
        refine_duration = time.time() - refine_start_time
        
        print(f"   优化后质量: {format_number(optimized_value)} 秒")
        print(f"   优化用时: {refine_duration:.2f}秒")
        print(f"   改进幅度: +{format_number(optimized_value - seeds[i][0])} 秒")
        
        # 更新最佳解
        if optimized_value > best_total_time:
            best_total_time = optimized_value
            best_solution_pack = (optimized_explosion_time, optimized_angle, optimized_distance)
    
    # 第三阶段: 精确评估最优解
    print(f"\n🎯 第三阶段: 最优解精确评估")
    
    explosion_time, angle, distance = best_solution_pack
    total_time, intervals, explosion_time, angle, distance, explosion_x, explosion_y = \
        OptimizationEngine.evaluate_solution_detailed(
            explosion_time, angle, distance, 
            scan_step=OptimizationConfig.FINE_SCAN_DT, 
            tolerance=OptimizationConfig.FINE_TOLERANCE)
    
    print(f"✅ 精确评估完成!")
    print(f"🏆 精确评估结果: {format_number(total_time)} 秒")
    
    # 第四阶段: 超精细验证
    print(f"\n🔬 第四阶段: 超精细验证")
    
    ultra_total_time, ultra_intervals, _, _, _, ultra_explosion_x, ultra_explosion_y = \
        OptimizationEngine.evaluate_solution_ultra_fine(explosion_time, angle, distance)
    
    print(f"✅ 超精细验证完成!")
    print(f"🏆 最终最优解: {format_number(ultra_total_time)} 秒")
    print(f"📊 精度提升: {format_number(ultra_total_time - total_time)} 秒")
    
    # 使用超精细结果作为最终结果
    total_time, intervals = ultra_total_time, ultra_intervals
    explosion_x, explosion_y = ultra_explosion_x, ultra_explosion_y
    
    # 输出详细结果
    ResultProcessor.print_optimization_results(
        total_time, intervals, explosion_time, angle, distance, explosion_x, explosion_y)
    
    # 性能统计
    total_duration = time.time() - start_time
    print("📈 性能统计:")
    print(f"   • 目标函数调用次数: {g_function_calls:,}")
    print(f"   • 总计算时间: {total_duration:.2f} 秒")
    print(f"   • 平均每次调用: {total_duration/g_function_calls*1000:.2f} 毫秒")
    print("=" * 70)
    print("🎉 优化完成!")

if __name__ == "__main__":
    main()
