#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q1_solver.py - 问题1求解器
=====================================

问题描述：
- 单架无人机FY1以固定参数执行烟幕干扰任务
- FY1以120 m/s朝向假目标方向飞行
- 受领任务1.5s后投放烟幕弹，间隔3.6s后起爆
- 计算对导弹M1的有效遮蔽时间

核心算法：
1. 建立三维几何模型（导弹轨迹、无人机轨迹、烟幕云团）
2. 使用LOS（视线）阻断判定：烟幕球体是否遮挡导弹到真目标的视线
3. 对真目标圆柱体进行采样，计算最难遮蔽的点
4. 通过数值方法求解有效遮蔽时间区间

输出：
- result_q1.xlsx：标准格式结果文件
- 可视化图表：三维场景图、俯视图、时间分析图
"""

import numpy as np, math, pandas as pd, os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

# ==================== 图表配置 ====================
# 设置中文字体，确保图表中的中文正常显示
plt.rcParams['font.sans-serif'] = ['STHeiti']  # macOS系统中文字体
plt.rcParams['axes.unicode_minus'] = False      # 解决负号显示问题

# ==================== 物理常量与场景参数 ====================
# 重力加速度
g = 9.8

# 烟幕参数
R_smoke = 10.0      # 烟幕球体半径 (m)
sink = 3.0          # 烟幕云团下沉速度 (m/s)
shield_T = 20.0     # 烟幕有效持续时间 (s)

# 导弹参数
v_missile = 300.0                           # 导弹飞行速度 (m/s)
M0 = np.array([20000.0, 0.0, 2000.0])      # 导弹M1初始位置 (x,y,z)

# 无人机参数
U0 = np.array([17800.0, 0.0, 1800.0])      # 无人机FY1初始位置 (x,y,z)
v_uav = 120.0                               # 无人机飞行速度 (m/s)
t_drop = 1.5                                # 受领任务后投放延时 (s)
delay_after_drop = 3.6                      # 投放后起爆延时 (s)
t_explode = t_drop + delay_after_drop       # 总起爆时间

# 真目标圆柱体参数
x_c, y_c = 0.0, 200.0      # 圆柱体底面圆心坐标
r_true = 7.0               # 圆柱体半径 (m)
z0_true = 0.0              # 圆柱体底面高度 (m)
h_true = 10.0              # 圆柱体高度 (m)

# ==================== 轨迹计算函数 ====================
# 计算导弹单位方向向量（指向假目标原点）
uM = -M0 / np.linalg.norm(M0)

# 计算导弹撞击假目标的时间
t_hit = np.linalg.norm(M0 - np.array([0.0,0.0,0.0])) / v_missile

def missile_pos(t):
    """
    计算t时刻导弹的位置
    
    参数:
        t: 时间 (s)
    
    返回:
        导弹位置向量 [x, y, z]
    
    原理:
        导弹从M0点出发，以恒定速度v_missile沿直线飞向假目标原点
        位置 = 初始位置 + 速度向量 × 时间
    """
    return M0 + v_missile * uM * t

def uav_unit_dir_to_origin(U0=U0):
    """
    计算无人机朝向假目标的单位方向向量
    
    参数:
        U0: 无人机初始位置
    
    返回:
        水平面内指向原点的单位向量
    
    原理:
        只考虑水平方向(x,y)，忽略高度差
        方向向量 = 目标位置 - 当前位置
    """
    horiz = np.array([-U0[0], -U0[1], 0.0])  # 指向原点的水平向量
    return horiz / np.linalg.norm(horiz)      # 归一化为单位向量

# 计算无人机飞行方向
Udir = uav_unit_dir_to_origin()

def explosion_center():
    """
    计算烟幕弹的起爆中心位置
    
    返回:
        起爆点坐标 [x, y, z]
    
    计算过程:
        1. 计算投放点：无人机在t_drop时刻的位置
        2. 计算起爆点：考虑投放后的抛物线运动
           - 水平方向：继续按无人机速度飞行
           - 垂直方向：受重力影响做自由落体运动
    """
    # 步骤1：计算投放时无人机位置
    r_drop = U0 + v_uav * Udir * t_drop
    
    # 步骤2：计算从投放到起爆的时间间隔
    dt = t_explode - t_drop  # = delay_after_drop = 3.6s
    
    # 步骤3：计算起爆点位置（抛物线运动）
    r_expl = r_drop + v_uav * Udir * dt + np.array([0.0, 0.0, -0.5*g*dt*dt])
    #        投放点   +   水平位移        +        垂直下降
    
    return r_expl

# 计算并存储起爆中心位置
R_expl = explosion_center()

def cloud_center(t):
    """
    计算t时刻烟幕云团的中心位置
    
    参数:
        t: 时间 (s)，应该 >= t_explode
    
    返回:
        云团中心坐标 [x, y, z]
    
    原理:
        起爆后云团以恒定速度sink向下沉降
        位置 = 起爆位置 + 下沉位移
    """
    return R_expl + np.array([0.0, 0.0, -sink*(t - t_explode)])

# ==================== 目标采样与遮蔽判定 ====================
def cylinder_samples(n_th=36, n_z=5, face_rs=(0.0, 3.5, None)):
    """
    对真目标圆柱体进行采样，生成代表性点集
    
    参数:
        n_th: 圆周方向采样点数
        n_z: 高度方向采样点数  
        face_rs: 顶底面径向采样半径列表
    
    返回:
        采样点数组，形状为 (N, 3)
    
    采样策略:
        1. 圆柱面采样：在不同高度的圆周上均匀采样
        2. 顶底面采样：在不同半径的圆周上采样
        目的：确保能检测到最难遮蔽的关键点
    """
    if face_rs[-1] is None:
        face_rs = list(face_rs[:-1]) + [r_true]
    
    pts = []
    
    # 生成角度和高度采样点
    thetas = np.linspace(0, 2*math.pi, n_th, endpoint=False)
    zs = np.linspace(z0_true, z0_true+h_true, n_z)
    
    # 1. 圆柱面采样
    for z in zs:
        for th in thetas:
            x = x_c + r_true * math.cos(th)
            y = y_c + r_true * math.sin(th)
            pts.append((x, y, z))
    
    # 2. 顶底面采样
    for z_fixed in (z0_true, z0_true+h_true):  # 底面和顶面
        for r in face_rs:                       # 不同半径
            for th in thetas:                   # 圆周采样
                x = x_c + r * math.cos(th)
                y = y_c + r * math.sin(th)
                pts.append((x, y, z_fixed))
    
    return np.array(pts, dtype=float)

# 生成目标采样点集（高密度采样以提高精度）
SAMPLES = cylinder_samples(n_th=48, n_z=6, face_rs=(0.0, r_true/2, None))

def point_to_segment_dist(P, A, B):
    """
    计算点P到线段AB的最短距离
    
    参数:
        P: 查询点坐标
        A, B: 线段端点坐标
    
    返回:
        最短距离值
    
    算法原理:
        1. 计算点P在线段AB上的投影参数t
        2. 将t限制在[0,1]范围内（确保投影点在线段上）
        3. 计算P到投影点的距离
    """
    AB = B - A
    denom = float(np.dot(AB, AB))
    
    # 处理退化情况：A和B重合
    if denom == 0.0:
        return float(np.linalg.norm(P - A))
    
    # 计算投影参数
    tt = float(np.dot(P - A, AB) / denom)
    
    # 限制在线段范围内
    tt = 0.0 if tt < 0.0 else (1.0 if tt > 1.0 else tt)
    
    # 计算距离
    return float(np.linalg.norm(P - (A + tt*AB)))

def make_g_cylinder_all(samples):
    """
    构造遮蔽判定函数g(t)
    
    参数:
        samples: 目标采样点数组
    
    返回:
        函数g(t)，当g(t) <= 0时表示有效遮蔽
    
    算法原理:
        1. 对于每个采样点Q，计算烟幕中心P到"导弹-Q"线段的距离
        2. 找出所有距离中的最大值（最难遮蔽的点）
        3. g(t) = 最大距离 - 烟幕半径
        4. 当g(t) <= 0时，说明烟幕能遮蔽所有采样点
    """
    samples = np.asarray(samples, dtype=float)
    
    def g(t):
        P = cloud_center(t)        # t时刻烟幕中心位置
        A = missile_pos(t)         # t时刻导弹位置
        
        dmax = 0.0
        # 遍历所有采样点，找出最难遮蔽的点
        for Q in samples:
            d = point_to_segment_dist(P, A, Q)  # 烟幕中心到"导弹-目标点"线段的距离
            if d > dmax: 
                dmax = d
        
        return dmax - R_smoke  # 距离 - 烟幕半径
    
    return g

def intervals_from_g(gfunc, t0, t1, dt_scan=0.01, tol=1e-4):
    """
    根据遮蔽函数g(t)计算有效遮蔽时间区间
    
    参数:
        gfunc: 遮蔽判定函数
        t0, t1: 时间范围
        dt_scan: 粗扫描步长
        tol: 二分法精度
    
    返回:
        (总遮蔽时间, 遮蔽区间列表)
    
    算法流程:
        1. 粗扫描：以dt_scan步长计算g(t)值
        2. 根号查找：用二分法精确定位g(t)=0的根
        3. 区间构造：根据符号变化确定遮蔽区间
    """
    # 步骤1：粗扫描
    times = [t0]
    cur = t0
    while cur < t1:
        cur = min(cur + dt_scan, t1)
        times.append(cur)
    
    vals = [gfunc(tt) for tt in times]
    
    # 步骤2：二分法求根
    def bisect(a, b, fa, fb):
        """二分法求解g(t)=0的根"""
        if fa == 0.0: return a
        if fb == 0.0: return b
        
        lo, hi, flo, fhi = a, b, fa, fb
        for _ in range(64):  # 最多64次迭代
            mid = 0.5*(lo + hi)
            fm = gfunc(mid)
            
            if abs(fm) < 1e-12 or (hi - lo) < tol:
                return mid
            
            if flo * fm <= 0.0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        
        return 0.5*(lo + hi)
    
    # 查找所有根
    roots = []
    for i in range(1, len(times)):
        a, b = times[i-1], times[i]
        fa, fb = vals[i-1], vals[i]
        
        if fa == 0.0:
            roots.append(a)
        if fa * fb < 0.0:  # 符号变化，存在根
            roots.append(bisect(a, b, fa, fb))
    
    roots = sorted(set(roots))
    
    # 步骤3：构造遮蔽区间
    intervals = []
    inside = (vals[0] <= 0.0)  # 初始状态是否在遮蔽区间内
    start = t0 if inside else None
    
    for r in roots:
        if inside:
            # 结束当前区间
            intervals.append((start, r))
            inside = False
            start = None
        else:
            # 开始新区间
            inside = True
            start = r
    
    # 处理最后一个区间
    if inside:
        intervals.append((start, t1))
    
    # 计算总遮蔽时间
    total = sum(b - a for a, b in intervals)
    
    return total, intervals

# ==================== 可视化函数 ====================
def create_visualizations(intervals, total):
    """
    创建Q1问题的可视化图表
    
    参数:
        intervals: 遮蔽时间区间列表
        total: 总遮蔽时间
    
    生成图表:
        1. 三维场景图：显示导弹、无人机轨迹和烟幕位置
        2. 俯视图：XY平面投影，便于理解空间关系
        3. 时间分析图：遮蔽函数变化和时间轴分析
    """
    # 确保输出目录存在
    os.makedirs("./output", exist_ok=True)
    
    # ========== 图表1：三维场景图 ==========
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制导弹轨迹
    t_trajectory = np.linspace(0, t_hit, 100)
    missile_trajectory = np.array([missile_pos(t) for t in t_trajectory])
    ax.plot(missile_trajectory[:, 0], missile_trajectory[:, 1], missile_trajectory[:, 2], 
            'r-', linewidth=3, label='导弹M1轨迹')
    
    # 绘制无人机轨迹
    t_uav = np.linspace(0, 10, 50)
    uav_trajectory = np.array([U0 + v_uav * Udir * t for t in t_uav])
    ax.plot(uav_trajectory[:, 0], uav_trajectory[:, 1], uav_trajectory[:, 2], 
            'b-', linewidth=2, label='无人机FY1轨迹')
    
    # 标记关键点
    ax.scatter(*M0, color='red', s=100, label='导弹初始位置M1')
    ax.scatter(*U0, color='blue', s=100, label='无人机初始位置FY1')
    ax.scatter(0, 0, 0, color='black', s=100, marker='s', label='假目标')
    ax.scatter(0, 200, 0, color='green', s=100, marker='^', label='真目标')
    
    # 绘制投放点和起爆点
    drop_point = U0 + v_uav * Udir * t_drop
    ax.scatter(*drop_point, color='orange', s=150, marker='*', label='烟幕弹投放点')
    ax.scatter(*R_expl, color='purple', s=150, marker='o', label='烟幕弹起爆点')
    
    # 绘制烟幕球体（在起爆时刻的位置）
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = R_smoke * np.outer(np.cos(u), np.sin(v)) + R_expl[0]
    y_sphere = R_smoke * np.outer(np.sin(u), np.sin(v)) + R_expl[1]
    z_sphere = R_smoke * np.outer(np.ones(np.size(u)), np.cos(v)) + R_expl[2]
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='gray')
    
    # 绘制真目标圆柱体
    theta = np.linspace(0, 2*np.pi, 20)
    z_cyl = np.linspace(z0_true, z0_true + h_true, 10)
    theta_mesh, z_mesh = np.meshgrid(theta, z_cyl)
    x_cyl = x_c + r_true * np.cos(theta_mesh)
    y_cyl = y_c + r_true * np.sin(theta_mesh)
    ax.plot_surface(x_cyl, y_cyl, z_mesh, alpha=0.4, color='green')
    
    ax.set_xlabel('X (米)')
    ax.set_ylabel('Y (米)')
    ax.set_zlabel('Z (米)')
    ax.set_title('Q1: 单无人机单烟幕弹干扰场景三维图', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('./output/q1_3d_scenario.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图表2：俯视图（XY平面） ==========
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制轨迹投影
    ax.plot(missile_trajectory[:, 0], missile_trajectory[:, 1], 'r-', linewidth=3, label='导弹M1轨迹')
    ax.plot(uav_trajectory[:, 0], uav_trajectory[:, 1], 'b-', linewidth=2, label='无人机FY1轨迹')
    
    # 标记关键点
    ax.scatter(M0[0], M0[1], color='red', s=100, label='导弹初始位置M1')
    ax.scatter(U0[0], U0[1], color='blue', s=100, label='无人机初始位置FY1')
    ax.scatter(0, 0, color='black', s=100, marker='s', label='假目标')
    ax.scatter(0, 200, color='green', s=100, marker='^', label='真目标')
    ax.scatter(drop_point[0], drop_point[1], color='orange', s=150, marker='*', label='烟幕弹投放点')
    ax.scatter(R_expl[0], R_expl[1], color='purple', s=150, marker='o', label='烟幕弹起爆点')
    
    # 绘制真目标圆柱体俯视图
    circle = patches.Circle((x_c, y_c), r_true, linewidth=2, edgecolor='green', 
                          facecolor='lightgreen', alpha=0.3, label='真目标保护区')
    ax.add_patch(circle)
    
    # 绘制烟幕覆盖区域
    smoke_circle = patches.Circle((R_expl[0], R_expl[1]), R_smoke, linewidth=2, 
                                edgecolor='purple', facecolor='gray', alpha=0.3, 
                                label='烟幕覆盖区域')
    ax.add_patch(smoke_circle)
    
    ax.set_xlabel('X (米)')
    ax.set_ylabel('Y (米)')
    ax.set_title('Q1: 单无人机单烟幕弹干扰场景俯视图', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('./output/q1_top_view.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== 图表3：遮蔽时间分析图 ==========
    if intervals:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 上图：遮蔽函数随时间变化
        t_analysis = np.linspace(t_explode, min(t_explode + shield_T, t_hit), 1000)
        g_all = make_g_cylinder_all(SAMPLES)
        g_values = [g_all(t) for t in t_analysis]
        
        ax1.plot(t_analysis, g_values, 'b-', linewidth=2, label='遮蔽函数g(t)')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='遮蔽阈值')
        ax1.fill_between(t_analysis, g_values, 0, where=np.array(g_values) <= 0, 
                        alpha=0.3, color='green', label='有效遮蔽区间')
        
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('遮蔽函数值')
        ax1.set_title('Q1: 遮蔽效果随时间变化', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True)
        
        # 下图：遮蔽区间可视化
        ax2.barh(0, t_hit, height=0.3, color='lightcoral', alpha=0.7, label='导弹飞行时间')
        ax2.barh(0.5, shield_T, left=t_explode, height=0.3, color='lightblue', alpha=0.7, 
                label='烟幕持续时间')
        
        for i, (start, end) in enumerate(intervals):
            ax2.barh(1, end-start, left=start, height=0.3, color='green', alpha=0.8, 
                    label='有效遮蔽时间' if i == 0 else "")
        
        ax2.set_xlabel('时间 (秒)')
        ax2.set_yticks([0, 0.5, 1])
        ax2.set_yticklabels(['导弹飞行', '烟幕持续', '有效遮蔽'])
        ax2.set_title('Q1: 时间轴分析', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, axis='x')
        
        plt.tight_layout()
        plt.savefig('./output/q1_time_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("✓ Q1可视化图表已保存到output/目录")

# ==================== 主程序 ====================
def main():
    """
    Q1问题主求解流程
    
    执行步骤:
        1. 参数初始化和场景设置
        2. 构造遮蔽判定函数
        3. 计算有效遮蔽时间区间
        4. 生成可视化图表
        5. 保存结果到Excel文件
    """
    print("=" * 60)
    print("Q1: 单无人机单烟幕弹对M1导弹干扰问题求解")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs("./output", exist_ok=True)
    
    # ========== 步骤1：设置计算时间范围 ==========
    # 有效计算区间：从起爆开始，到烟幕失效或导弹撞击为止
    t0 = max(t_explode, 0.0)                        # 起始时间：起爆时刻
    t1 = min(t_explode + shield_T, t_hit)           # 结束时间：烟幕失效或导弹撞击
    
    # ========== 步骤2：构造遮蔽判定函数 ==========
    g_all = make_g_cylinder_all(SAMPLES)
    
    # ========== 步骤3：计算有效遮蔽区间 ==========
    total, intervals = intervals_from_g(g_all, t0, t1, dt_scan=0.01, tol=1e-4)
    
    # ========== 步骤4：输出计算结果 ==========
    print("\n📊 计算结果:")
    print(f"  起爆时间: {t_explode:.2f} 秒")
    print(f"  起爆中心: ({R_expl[0]:.1f}, {R_expl[1]:.1f}, {R_expl[2]:.1f}) 米")
    print(f"  导弹撞击时间: {t_hit:.2f} 秒")
    print(f"  总遮蔽时间: {total:.6f} 秒")
    print(f"  遮蔽区间: {[(round(a,3), round(b,3)) for a,b in intervals]}")
    
    # ========== 步骤5：诊断分析 ==========
    # 找出最难遮蔽的点，用于验证算法正确性
    if intervals:
        tmid = 0.5*(intervals[0][0] + intervals[0][1])  # 取第一个区间的中点
        P = cloud_center(tmid)                          # 该时刻烟幕中心
        A = missile_pos(tmid)                           # 该时刻导弹位置
        
        # 计算所有采样点到"导弹-目标"线段的距离
        dists = [point_to_segment_dist(P, A, Q) for Q in SAMPLES]
        idx = int(np.argmax(dists))                     # 找出最远的点
        
        print(f"  最难遮蔽点: {tuple(np.round(SAMPLES[idx],1))} 米")
        print(f"  距离-半径: {round(dists[idx]-R_smoke,3)} 米")
    
    # ========== 步骤6：生成可视化图表 ==========
    print("\n🎨 生成可视化图表...")
    create_visualizations(intervals, total)
    
    # ========== 步骤7：保存Excel结果文件 ==========
    # 计算投放点位置
    drop_point = U0 + v_uav * Udir * t_drop
    
    # 按照题目要求的格式保存result1
    rows = [{
        '无人机运动方向': round(math.degrees(math.atan2(Udir[1], Udir[0])) % 360, 1),
        '无人机运动速度 (m/s)': v_uav,
        '烟幕干扰弹编号': 1,
        '烟幕干扰弹投放点的x坐标 (m)': round(drop_point[0], 1),
        '烟幕干扰弹投放点的y坐标 (m)': round(drop_point[1], 1),
        '烟幕干扰弹投放点的z坐标 (m)': round(drop_point[2], 1),
        '烟幕干扰弹起爆点的x坐标 (m)': round(R_expl[0], 1),
        '烟幕干扰弹起爆点的y坐标 (m)': round(R_expl[1], 1),
        '烟幕干扰弹起爆点的z坐标 (m)': round(R_expl[2], 1),
        '有效干扰时长 (s)': round(total, 6)
    }]
    
    df = pd.DataFrame(rows)
    
    # 保存到output目录 - Q1对应的是q1_data.xlsx
    output_file = "./output/q1_data.xlsx"
    df.to_excel(output_file, index=False)
    print(f"✓ 结果已保存到: {output_file}")
    
    # 同时保存一个详细版本用于调试和进一步分析
    detailed_rows = [{
        'scenario': 'Q1_fixed',
        'uav_id': 'FY1',
        'uav_direction_deg': round(math.degrees(math.atan2(Udir[1], Udir[0])) % 360, 1),
        'uav_speed_m_s': v_uav,
        'drop_time_s': t_drop,
        'explode_delay_s': delay_after_drop,
        'explode_time_s': t_explode,
        'drop_x': round(drop_point[0], 1),
        'drop_y': round(drop_point[1], 1),
        'drop_z': round(drop_point[2], 1),
        'explode_x': round(R_expl[0], 1),
        'explode_y': round(R_expl[1], 1),
        'explode_z': round(R_expl[2], 1),
        'total_coverage_time_s': round(total, 6),
        'coverage_intervals': str([(round(a,3), round(b,3)) for a,b in intervals])
    }]
    
    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_excel("./output/q1_detailed_results.xlsx", index=False)
    
    # ========== 步骤8：总结输出 ==========
    print("\n📁 生成的文件:")
    print("  - output/q1_data.xlsx (Q1标准结果)")
    print("  - output/q1_detailed_results.xlsx (详细分析数据)")
    print("  - output/q1_3d_scenario.png (三维场景图)")
    print("  - output/q1_top_view.png (俯视图)")
    print("  - output/q1_time_analysis.png (时间分析图)")
    print("\n✅ Q1问题求解完成！")

# ==================== 程序入口 ====================
if __name__ == "__main__":
    main()
