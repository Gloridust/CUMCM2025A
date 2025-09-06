# -*- coding: utf-8 -*-
"""
问题4（Q4）求解器：三架无人机（FY1、FY2、FY3）各投放一枚烟幕干扰弹，
通过粒子群优化（PSO）搜索无人机航向、速度、投放时刻与引信延时，
以最大化对真目标的联合有效遮蔽时长（允许不连续）。

说明：
- 模型假设导弹以300 m/s的恒速，从给定初始点沿直线指向假目标原点(0,0,0)；
- 烟幕云团起爆后半径R0=10 m，持续shield_T=20 s，云团中心以sink=3 m/s下沉；
- 无人机接令后航向瞬时可调，随后以70~140 m/s等高度匀速直线飞行。
"""

import numpy as np
import pandas as pd
import random
from math import cos, sin
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# ============ 物理/场景 ============
g = 9.8
R0 = 10.0          # 烟幕半径 (m)
sink = 3.0         # 云团中心下沉速度 (m/s)
shield_T = 20.0    # 起爆后有效时间 (s)

# 真目标
T  = np.array([0.0, 200.0, 0.0], dtype=float)

# 导弹：从 M0 出发，恒速 v_missile 朝假目标圆心(0,0,0)
M0 = np.array([20000.0, 0.0, 2000.0], dtype=float)
v_missile = 300.0

def missile_direction_unit() -> np.ndarray:
    """导弹朝向单位向量（指向假目标原点）。"""
    u = -M0
    return u / np.linalg.norm(u)

MISSILE_DIR = missile_direction_unit()

def missile_position(t: float) -> np.ndarray:
    """返回 t 秒时刻的导弹位置。"""
    return M0 + v_missile * MISSILE_DIR * t

# 三架无人机初始（如与你不同请改这里）
FY1 = np.array([17800.0,     0.0, 1800.0], dtype=float)
FY2 = np.array([12000.0,  1400.0, 1400.0], dtype=float)
FY3 = np.array([ 6000.0, -3000.0,  700.0], dtype=float)
U0_list = [FY1, FY2, FY3]
Z0_list = [FY1[2], FY2[2], FY3[2]]

# 速度范围
V_MIN, V_MAX = 70.0, 140.0

# ============ 目标函数权重 ============
LAMBDA_PAIR   = 0.0   # 两两重叠惩罚（可按需>0）
LAMBDA_TRIPLE = 0.0   # 三重重叠惩罚（可按需>0）
MU_FRAGMENT   = 0.25  # 并集分段惩罚：每多一段扣 MU_FRAGMENT 秒（鼓励连贯）
MIN_SINGLE    = 0.5   # 每枚至少贡献这么多秒（软惩罚阈值）
ALPHA_EACH    = 5.0   # 不足阈值时的扣分强度

# ============ 精度/数值参数 ============
DT_SCAN    = 0.08
BISECT_TOL = 8e-5
SEED       = 12345
random.seed(SEED); np.random.seed(SEED)

# ============ PSO 参数 ============
SWARM_SIZE = 80
ITER_MAX   = 120
W_INERTIA  = 0.78
C1         = 1.6
C2         = 1.6
VMAX_FRAC  = 0.3
REPORT_EVERY = 5

# ============ 时间/延时约束 ============
TMAX        = 20.0    # 投放上限（D ≤ TMAX）
E_MAX_ABS   = 20.0    # 起爆上限（E ≤ E_MAX_ABS）
DELTA_MIN   = 0.0     # 允许引信延时为 0

# ============ 工具 ============
def fmt3(x: float) -> float:
    """将数值格式化为三位小数（float）。"""
    return float(f"{float(x):.3f}")


def heading_to_unit_vector(theta_deg: float) -> np.ndarray:
    """将航向角(度)转换为单位方向向量（xy 平面；x为正向、逆时针为正）。"""
    th = np.deg2rad(theta_deg)
    return np.array([cos(th), sin(th), 0.0], dtype=float)


def point_to_segment_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """点 P 到线段 AB 的最短距离。"""
    AB = B - A
    d2 = float(np.dot(AB, AB))
    if d2 == 0.0:
        return float(np.linalg.norm(P - A))
    tt = float(np.dot(P - A, AB) / d2)
    tt = max(0.0, min(1.0, tt))
    return float(np.linalg.norm(P - (A + tt*AB)))


def shielding_intervals_center(t_expl: float, x_e: float, y_e: float, z_e: float,
                               dt_scan: float = DT_SCAN, tol: float = BISECT_TOL) -> Tuple[float, List[Tuple[float, float]]]:
    """计算云团中心在[t_expl, t_expl+shield_T]内对“导弹-真目标线段”的遮蔽区间。
    条件：中心到线段(导弹(t)->T)的最短距离 ≤ R0 视为遮蔽。
    返回：(总遮蔽时长, 遮蔽区间列表)
    """
    t0, t1 = t_expl, t_expl + shield_T
    if t1 <= t0:
        return 0.0, []

    def C(t: float) -> np.ndarray:
        return np.array([x_e, y_e, z_e - sink*(t - t_expl)], dtype=float)

    def g(t: float) -> float:
        return point_to_segment_dist(C(t), missile_position(t), T) - R0

    # 粗扫
    ts = [t0]
    cur = t0
    while cur < t1:
        cur = min(cur + dt_scan, t1)
        ts.append(cur)
    vs = [g(t) for t in ts]
    if min(vs) > 8.0:
        return 0.0, []

    # 二分查找边界
    def root(a: float, b: float, fa: float, fb: float) -> float:
        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        lo, hi, flo, fhi = a, b, fa, fb
        for _ in range(64):
            mid = 0.5*(lo+hi)
            fm = g(mid)
            if abs(fm) < 1e-12 or (hi-lo) < tol:
                return mid
            if flo*fm <= 0.0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        return 0.5*(lo+hi)

    roots: List[float] = []
    for i in range(1, len(ts)):
        a, b = ts[i-1], ts[i]
        fa, fb = vs[i-1], vs[i]
        if fa == 0.0:
            roots.append(a)
        if fa*fb < 0.0:
            roots.append(root(a, b, fa, fb))
    roots = sorted(set(roots))

    # 拼接遮蔽区间
    ints: List[Tuple[float, float]] = []
    inside = (vs[0] <= 0.0)
    start = t0 if inside else None
    for r in roots:
        if inside:
            ints.append((start, r))
            inside = False
            start = None
        else:
            inside = True
            start = r
    if inside:
        ints.append((start, t1))
    return sum(b-a for a, b in ints), ints


def merge_intervals(xs: List[Tuple[float, float]]) -> Tuple[float, List[Tuple[float, float]]]:
    """将若干区间合并，并返回(总长度, 合并后区间)。"""
    if not xs:
        return 0.0, []
    xs = sorted(xs, key=lambda x: x[0])
    out = [list(xs[0])]
    for a, b in xs[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    out2 = [(a, b) for a, b in out]
    return sum(b-a for a, b in out2), out2


def intersect_two(i1: List[Tuple[float, float]], i2: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """两集合的区间交集（均为不重叠有序区间）。"""
    i = j = 0
    out: List[Tuple[float, float]] = []
    while i < len(i1) and j < len(i2):
        a1, b1 = i1[i]
        a2, b2 = i2[j]
        a = max(a1, a2)
        b = min(b1, b2)
        if a < b:
            out.append((a, b))
        if b1 < b2:
            i += 1
        else:
            j += 1
    return out


def intersect_three(i1: List[Tuple[float, float]], i2: List[Tuple[float, float]], i3: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """三集合的区间交集。"""
    return intersect_two(intersect_two(i1, i2), i3)


def total_len(ints: List[Tuple[float, float]]) -> float:
    return sum(b-a for a, b in ints)


def objective(ints1: List[Tuple[float, float]],
              ints2: List[Tuple[float, float]],
              ints3: List[Tuple[float, float]],
              lpair: float = LAMBDA_PAIR,
              ltrip: float = LAMBDA_TRIPLE,
              mu: float = MU_FRAGMENT) -> Tuple[float, float, List[Tuple[float, float]], Tuple[float, float, int]]:
    """目标函数：联合遮蔽时长 + 连贯性，减去重叠惩罚。"""
    U, Uints = merge_intervals(ints1 + ints2 + ints3)
    i12 = intersect_two(ints1, ints2)
    i13 = intersect_two(ints1, ints3)
    i23 = intersect_two(ints2, ints3)
    i123 = intersect_three(ints1, ints2, ints3)
    ov_pair = total_len(i12) + total_len(i13) + total_len(i23)
    ov_trpl = total_len(i123)
    frag    = len(Uints)
    J = U - lpair*ov_pair - ltrip*ov_trpl - mu*max(0, frag-1)
    return J, U, Uints, (ov_pair, ov_trpl, frag)

# ============ 评估（Δ 为变量，可为 0；含软惩罚） ============
# 粒子 x = [v1, th1, D1, Δ1,  v2, th2, D2, Δ2,  v3, th3, D3, Δ3]

def valid_and_eval(x: np.ndarray) -> Optional[Tuple[float, float, List[Tuple[float, float]], Tuple[float, float, int]]]:
    """给定12维决策变量，返回(目标J, 并集U, 并集区间, (两两重叠, 三重重叠, 片段数))。若无效返回None。"""
    xs = list(x)
    ints_all: List[List[Tuple[float, float]]] = []
    E_list: List[float] = []
    totals: List[float] = []

    for i in range(3):
        v  = float(np.clip(xs[4*i + 0], V_MIN, V_MAX))
        th = float(xs[4*i + 1] % 360.0)
        D  = float(np.clip(xs[4*i + 2], 0.0, TMAX))

        # Δ 上下限：Δ ≥ 0；Δ ≤ min( sqrt(2 z0/g), E_MAX_ABS - D )
        z0 = Z0_list[i]
        delta_max_height = max(0.0, np.sqrt(2.0*z0/g) - 1e-9)  # 不落地
        delta_max_window = max(0.0, E_MAX_ABS - D)             # 不越窗
        delta_max = max(0.0, min(delta_max_height, delta_max_window))
        if delta_max < DELTA_MIN:   # 只有 D 太晚且窗不足时才会出现
            return None

        delta = float(np.clip(xs[4*i + 3], DELTA_MIN, delta_max))
        E = D + delta

        udir = heading_to_unit_vector(th)
        U0   = U0_list[i]
        x_e = U0[0] + v*udir[0]*E
        y_e = U0[1] + v*udir[1]*E
        z_e = z0 - 0.5*g*(delta**2)
        if z_e < 0.0:
            return None

        tot_i, ints_i = shielding_intervals_center(E, x_e, y_e, z_e)
        ints_all.append(ints_i)
        E_list.append(E)
        totals.append(tot_i)

    # 并集目标 + 分段惩罚
    J, U, Uints, Ov = objective(ints_all[0], ints_all[1], ints_all[2],
                                lpair=LAMBDA_PAIR, ltrip=LAMBDA_TRIPLE, mu=MU_FRAGMENT)

    # 软惩罚：若某弹时长 < MIN_SINGLE，则按缺口扣分
    shortfall = sum(max(0.0, MIN_SINGLE - t) for t in totals)
    J -= ALPHA_EACH * shortfall

    # 一致性（理论上不会触发，保险留着）
    if Uints and (Uints[0][0] + 1e-6 < min(E_list)):
        J -= 10.0

    return J, U, Uints, Ov

# ============ PSO ============
class PSO:
    def __init__(self, lb, ub, vmax, n_pop=SWARM_SIZE, iters=ITER_MAX,
                 w=W_INERTIA, c1=C1, c2=C2):
        self.lb = np.array(lb, float)
        self.ub = np.array(ub, float)
        self.vmax = np.array(vmax, float)
        self.n = len(lb)
        self.n_pop = n_pop
        self.iters = iters
        self.w, self.c1, self.c2 = w, c1, c2

    def init_pop(self):
        P = np.random.rand(self.n_pop, self.n)*(self.ub - self.lb) + self.lb
        V = np.zeros_like(P)
        # 初始修正：确保 Δ 不越窗/不落地
        for k in range(self.n_pop):
            for i in range(3):
                D = np.clip(P[k, 4*i + 2], 0.0, TMAX)
                z0 = Z0_list[i]
                dmax_h = max(0.0, np.sqrt(2.0*z0/g) - 1e-9)
                dmax_w = max(0.0, E_MAX_ABS - D)
                dmax = max(0.0, min(dmax_h, dmax_w))
                P[k, 4*i + 2] = D
                P[k, 4*i + 3] = np.clip(P[k, 4*i + 3], DELTA_MIN, dmax)
        return P, V

    def project(self, X):
        Y = np.clip(X, self.lb, self.ub)
        for k in range(Y.shape[0]):
            for i in range(3):
                Y[k, 4*i + 0] = np.clip(Y[k, 4*i + 0], V_MIN, V_MAX)  # v
                Y[k, 4*i + 1] = Y[k, 4*i + 1] % 360.0                 # θ
                D  = np.clip(Y[k, 4*i + 2], 0.0, TMAX)                # D
                Y[k, 4*i + 2] = D
                z0 = Z0_list[i]
                dmax_h = max(0.0, np.sqrt(2.0*z0/g) - 1e-9)
                dmax_w = max(0.0, E_MAX_ABS - D)
                dmax = max(0.0, min(dmax_h, dmax_w))
                Y[k, 4*i + 3] = np.clip(Y[k, 4*i + 3], DELTA_MIN, dmax)  # Δ
        return Y

    def run(self):
        P, V = self.init_pop()
        pbest = P.copy()
        pbest_val = np.full(self.n_pop, -1e18, dtype=float)
        pbest_aux = [None]*self.n_pop
        gbest = None; gbest_val = -1e18; gbest_aux = None

        # 初评估
        for k in range(self.n_pop):
            eva = valid_and_eval(pbest[k])
            if eva is None:
                continue
            J, U, Uints, Ov = eva
            pbest_val[k] = J
            pbest_aux[k]  = (U, Uints, Ov)
            if J > gbest_val:
                gbest_val, gbest, gbest_aux = J, pbest[k].copy(), (U, Uints, Ov)

        # 迭代
        for it in range(1, self.iters+1):
            r1 = np.random.rand(self.n_pop, self.n)
            r2 = np.random.rand(self.n_pop, self.n)
            G  = np.tile(gbest, (self.n_pop,1))
            V  = self.w*V + self.c1*r1*(pbest - P) + self.c2*r2*(G - P)
            V  = np.clip(V, -self.vmax, self.vmax)
            P  = self.project(P + V)

            for k in range(self.n_pop):
                eva = valid_and_eval(P[k])
                if eva is None:
                    continue
                J, U, Uints, Ov = eva
                if (J > pbest_val[k]):
                    pbest_val[k] = J
                    pbest[k] = P[k].copy()
                    pbest_aux[k] = (U, Uints, Ov)
                    if J > gbest_val:
                        gbest_val, gbest, gbest_aux = J, P[k].copy(), (U, Uints, Ov)

            if (it % REPORT_EVERY) == 0 or it == 1:
                U, Uints, Ov = gbest_aux
                print(f"[迭代 {it:3d}] 当前最优 J={fmt3(gbest_val)}，联合遮蔽时长 U={fmt3(U)}，"
                      f"两两重叠={fmt3(Ov[0])}，三重重叠={fmt3(Ov[1])}，片段数={int(Ov[2])}")

        return gbest_val, gbest, gbest_aux

# ============ 体检 & 导出 ============

def inspect_solution(x: np.ndarray, label: str = "方案体检") -> None:
    print(f"\n[{label}] (v, 航向角th, 投放时刻D, 引信延时Δ) ×3:")
    for i in range(3):
        v  = float(np.clip(x[4*i + 0], V_MIN, V_MAX))
        th = float(x[4*i + 1] % 360.0)
        D  = float(np.clip(x[4*i + 2], 0.0, TMAX))
        z0 = Z0_list[i]
        dmax_h = max(0.0, np.sqrt(2.0*z0/g) - 1e-9)
        dmax_w = max(0.0, E_MAX_ABS - D)
        dmax = max(0.0, min(dmax_h, dmax_w))
        delta = float(np.clip(x[4*i + 3], DELTA_MIN, dmax))
        E = D + delta
        udir = heading_to_unit_vector(th)
        U0 = U0_list[i]
        x_e = U0[0] + v*udir[0]*E
        y_e = U0[1] + v*udir[1]*E
        z_e = z0 - 0.5*g*(delta**2)

        def C(t: float) -> np.ndarray:
            return np.array([x_e, y_e, z_e - sink*(t - E)], dtype=float)
        d_E = point_to_segment_dist(C(E), missile_position(E), T)
        ts = np.linspace(E, E + shield_T, 401)
        ds = [point_to_segment_dist(C(t), missile_position(t), T) for t in ts]

        print(f"  无人机#{i+1}: v={fmt3(v)} m/s, 航向角th={fmt3(th)}°, D={fmt3(D)} s, Δ={fmt3(delta)} s, E={fmt3(E)} s")
        print(f"           起爆高度z_e={fmt3(z_e)} m, 起爆时距线段={fmt3(d_E)} m, 窗内最小距离={fmt3(min(ds))} m")


def create_visualizations(best_params: List[float], best_fitness: float, best_intervals: List[Tuple[float, float]]) -> None:
    """创建Q4问题的可视化图表"""
    os.makedirs("./output", exist_ok=True)
    
    # 计算关键位置
    drone_trajectories = []
    drop_points = []
    explosion_points = []
    drone_names = ['FY1', 'FY2', 'FY3']
    colors = ['orange', 'cyan', 'magenta']
    
    for i in range(3):
        v = float(np.clip(best_params[4*i + 0], V_MIN, V_MAX))
        th = float(best_params[4*i + 1] % 360.0)
        D = float(np.clip(best_params[4*i + 2], 0.0, TMAX))
        z0 = Z0_list[i]
        dmax_h = max(0.0, np.sqrt(2.0*z0/g) - 1e-9)
        dmax_w = max(0.0, E_MAX_ABS - D)
        dmax = max(0.0, min(dmax_h, dmax_w))
        delta = float(np.clip(best_params[4*i + 3], DELTA_MIN, dmax))
        E = D + delta
        
        udir = heading_to_unit_vector(th)
        U0 = U0_list[i]
        
        # 计算投放点和起爆点
        r_drop = U0 + v * udir * D
        drop_points.append(r_drop)
        
        x_e = U0[0] + v * udir[0] * E
        y_e = U0[1] + v * udir[1] * E
        z_e = z0 - 0.5 * g * (delta ** 2)
        explosion_point = np.array([x_e, y_e, z_e])
        explosion_points.append(explosion_point)
        
        # 计算无人机轨迹
        t_traj = np.linspace(0, max(E * 1.2, 10), 50)
        traj = np.array([U0 + v * udir * t for t in t_traj])
        drone_trajectories.append(traj)
    
    # 1. 三维场景图
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制导弹轨迹
    missile_flight_time = np.linalg.norm(M0) / v_missile
    t_trajectory = np.linspace(0, missile_flight_time, 100)
    missile_trajectory = np.array([missile_position(t) for t in t_trajectory])
    ax.plot(missile_trajectory[:, 0], missile_trajectory[:, 1], missile_trajectory[:, 2], 
            'r-', linewidth=4, label='导弹M1轨迹')
    
    # 绘制三架无人机轨迹
    for i, (traj, drone_name, color) in enumerate(zip(drone_trajectories, drone_names, colors)):
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=color, linewidth=2, label=f'无人机{drone_name}轨迹')
    
    # 标记关键点
    ax.scatter(*M0, color='red', s=150, label='导弹初始位置M1')
    for i, (U0_i, drone_name, color) in enumerate(zip(U0_list, drone_names, colors)):
        ax.scatter(*U0_i, color=color, s=120, label=f'无人机{drone_name}初始位置')
    
    ax.scatter(0, 0, 0, color='black', s=120, marker='s', label='假目标')
    ax.scatter(*T, color='green', s=120, marker='^', label='真目标')
    
    # 绘制投放点、起爆点和烟幕球体
    for i, (drop_pt, expl_pt, drone_name, color) in enumerate(zip(drop_points, explosion_points, drone_names, colors)):
        ax.scatter(*drop_pt, color=color, s=200, marker='*', 
                  label=f'{drone_name}烟幕弹投放点')
        ax.scatter(*expl_pt, color=color, s=200, marker='o', 
                  label=f'{drone_name}烟幕弹起爆点')
        
        # 绘制烟幕球体
        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        x_sphere = R0 * np.outer(np.cos(u), np.sin(v)) + expl_pt[0]
        y_sphere = R0 * np.outer(np.sin(u), np.sin(v)) + expl_pt[1]
        z_sphere = R0 * np.outer(np.ones(np.size(u)), np.cos(v)) + expl_pt[2]
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color=color)
    
    # 绘制真目标圆柱体
    theta = np.linspace(0, 2*np.pi, 20)
    z_cyl = np.linspace(0, 10, 10)
    theta_mesh, z_mesh = np.meshgrid(theta, z_cyl)
    x_cyl = 7 * np.cos(theta_mesh)
    y_cyl = T[1] + 7 * np.sin(theta_mesh)
    ax.plot_surface(x_cyl, y_cyl, z_mesh, alpha=0.4, color='green')
    
    ax.set_xlabel('X (米)')
    ax.set_ylabel('Y (米)')
    ax.set_zlabel('Z (米)')
    ax.set_title('Q4: 三无人机协同烟幕干扰三维场景图', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('./output/q4_3d_scenario.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 俯视图（XY平面）
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # 绘制轨迹投影
    ax.plot(missile_trajectory[:, 0], missile_trajectory[:, 1], 'r-', linewidth=4, label='导弹M1轨迹')
    
    for i, (traj, drone_name, color) in enumerate(zip(drone_trajectories, drone_names, colors)):
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, label=f'无人机{drone_name}轨迹')
    
    # 标记关键点
    ax.scatter(M0[0], M0[1], color='red', s=150, label='导弹初始位置M1')
    for i, (U0_i, drone_name, color) in enumerate(zip(U0_list, drone_names, colors)):
        ax.scatter(U0_i[0], U0_i[1], color=color, s=120, label=f'无人机{drone_name}初始位置')
    
    ax.scatter(0, 0, color='black', s=120, marker='s', label='假目标')
    ax.scatter(T[0], T[1], color='green', s=120, marker='^', label='真目标')
    
    # 绘制投放点、起爆点和烟幕覆盖区域
    for i, (drop_pt, expl_pt, drone_name, color) in enumerate(zip(drop_points, explosion_points, drone_names, colors)):
        ax.scatter(drop_pt[0], drop_pt[1], color=color, s=200, marker='*', 
                  label=f'{drone_name}烟幕弹投放点')
        ax.scatter(expl_pt[0], expl_pt[1], color=color, s=200, marker='o', 
                  label=f'{drone_name}烟幕弹起爆点')
        
        # 绘制烟幕覆盖区域
        smoke_circle = patches.Circle((expl_pt[0], expl_pt[1]), R0, linewidth=2, 
                                    edgecolor=color, facecolor=color, alpha=0.2, 
                                    label=f'{drone_name}烟幕覆盖区域')
        ax.add_patch(smoke_circle)
    
    # 绘制真目标圆柱体俯视图
    circle_true = patches.Circle((T[0], T[1]), 7, linewidth=2, edgecolor='green', 
                               facecolor='lightgreen', alpha=0.3, label='真目标保护区')
    ax.add_patch(circle_true)
    
    # 绘制遮蔽时间段的视线
    if best_intervals:
        for i, (start, end) in enumerate(best_intervals[:3]):  # 最多显示3个时段
            mid_time = (start + end) / 2
            missile_pos_mid = missile_position(mid_time)
            ax.plot([missile_pos_mid[0], T[0]], [missile_pos_mid[1], T[1]], 
                   '--', linewidth=2, alpha=0.6, 
                   label=f'遮蔽时段{i+1}视线' if i < 3 else "")
    
    ax.set_xlabel('X (米)')
    ax.set_ylabel('Y (米)')
    ax.set_title('Q4: 三无人机协同烟幕干扰俯视图', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('./output/q4_top_view.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 参数优化分析图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 各无人机速度分布
    speeds = []
    for i in range(3):
        v = float(np.clip(best_params[4*i + 0], V_MIN, V_MAX))
        speeds.append(v)
    
    bars1 = ax1.bar(drone_names, speeds, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('速度 (m/s)')
    ax1.set_title('各无人机最优速度', fontweight='bold')
    ax1.grid(True, axis='y')
    
    # 在柱状图上添加数值标签
    for bar, speed in zip(bars1, speeds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{speed:.1f}m/s', ha='center', va='bottom', fontweight='bold')
    
    # 子图2: 各无人机航向角分布
    angles = []
    for i in range(3):
        th = float(best_params[4*i + 1] % 360.0)
        angles.append(th)
    
    bars2 = ax2.bar(drone_names, angles, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('航向角 (度)')
    ax2.set_title('各无人机最优航向角', fontweight='bold')
    ax2.grid(True, axis='y')
    
    for bar, angle in zip(bars2, angles):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{angle:.1f}°', ha='center', va='bottom', fontweight='bold')
    
    # 子图3: 投放时间和起爆时间
    drop_times = []
    explode_times = []
    for i in range(3):
        D = float(np.clip(best_params[4*i + 2], 0.0, TMAX))
        z0 = Z0_list[i]
        dmax_h = max(0.0, np.sqrt(2.0*z0/g) - 1e-9)
        dmax_w = max(0.0, E_MAX_ABS - D)
        dmax = max(0.0, min(dmax_h, dmax_w))
        delta = float(np.clip(best_params[4*i + 3], DELTA_MIN, dmax))
        E = D + delta
        drop_times.append(D)
        explode_times.append(E)
    
    x = np.arange(len(drone_names))
    width = 0.35
    
    bars3_1 = ax3.bar(x - width/2, drop_times, width, label='投放时间', color='lightblue', alpha=0.7)
    bars3_2 = ax3.bar(x + width/2, explode_times, width, label='起爆时间', color='lightcoral', alpha=0.7)
    
    ax3.set_ylabel('时间 (秒)')
    ax3.set_title('各无人机投放与起爆时间', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(drone_names)
    ax3.legend()
    ax3.grid(True, axis='y')
    
    # 子图4: 引信延时分布
    delays = []
    for i in range(3):
        D = float(np.clip(best_params[4*i + 2], 0.0, TMAX))
        z0 = Z0_list[i]
        dmax_h = max(0.0, np.sqrt(2.0*z0/g) - 1e-9)
        dmax_w = max(0.0, E_MAX_ABS - D)
        dmax = max(0.0, min(dmax_h, dmax_w))
        delta = float(np.clip(best_params[4*i + 3], DELTA_MIN, dmax))
        delays.append(delta)
    
    bars4 = ax4.bar(drone_names, delays, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('延时 (秒)')
    ax4.set_title('各无人机引信延时', fontweight='bold')
    ax4.grid(True, axis='y')
    
    for bar, delay in zip(bars4, delays):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{delay:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./output/q4_optimization_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Q4可视化图表已保存到output/目录")


def main():
    # 决策变量边界： v∈[70,140], th∈[0,360), D∈[0,TMAX], Δ∈[0, ?]（上限在投影时依 D,z0 确定）
    lb = [70, 0, 0, DELTA_MIN] * 3
    ub = [140, 360, TMAX, max(DELTA_MIN, E_MAX_ABS)] * 3
    vmax = [(ub[j]-lb[j]) * VMAX_FRAC for j in range(12)]

    pso = PSO(lb, ub, vmax)
    bestJ, x, (U, Uints, Ov) = pso.run()

    print("=" * 70)
    print("Q4: 三无人机协同烟幕干扰问题求解完成")
    print("=" * 70)
    
    print(f"\n📊 最优解结果:")
    print(f"  目标函数 J: {bestJ:.3f}")
    print(f"  联合遮蔽时长 U: {U:.3f} 秒")
    print(f"  联合区间: {[(fmt3(a), fmt3(b)) for a, b in Uints]}")
    print(f"  重叠统计：两两={fmt3(Ov[0])}, 三重={fmt3(Ov[1])}, 片段数={int(Ov[2])}")

    inspect_solution(x, "PSO 最优方案")
    
    # 生成可视化图表
    print("\n🎨 生成可视化图表...")
    create_visualizations(x, U, Uints)

    # 导出 Excel（保留原有详细字段 + 题面模板字段）
    print("\n💾 保存结果到Excel...")
    os.makedirs("./output", exist_ok=True)
    rows = []
    for i in range(3):
        v  = float(np.clip(x[4*i + 0], V_MIN, V_MAX))
        th = float(x[4*i + 1] % 360.0)
        D  = float(np.clip(x[4*i + 2], 0.0, TMAX))
        z0 = Z0_list[i]
        dmax_h = max(0.0, np.sqrt(2.0*z0/g) - 1e-9)
        dmax_w = max(0.0, E_MAX_ABS - D)
        dmax = max(0.0, min(dmax_h, dmax_w))
        delta = float(np.clip(x[4*i + 3], DELTA_MIN, dmax))
        E = D + delta
        udir = heading_to_unit_vector(th)
        U0 = U0_list[i]
        r_drop = U0 + v*udir*D
        x_e = U0[0] + v*udir[0]*E
        y_e = U0[1] + v*udir[1]*E
        z_e = z0 - 0.5*g*(delta**2)
        tot_i, _ = shielding_intervals_center(E, x_e, y_e, z_e)

        rows.append({
            '无人机编号': f'FY{i+1}',
            '无人机运动方向(度)': int(round(np.degrees(np.arctan2(udir[1], udir[0])) % 360)),
            '无人机运动速度(m/s)': fmt3(v),
            '烟幕干扰弹编号': i+1,
            '烟幕干扰弹投放点x(m)': fmt3(r_drop[0]),
            '烟幕干扰弹投放点y(m)': fmt3(r_drop[1]),
            '烟幕干扰弹投放点z(m)': fmt3(r_drop[2]),
            '烟幕干扰弹起爆点x(m)': fmt3(x_e),
            '烟幕干扰弹起爆点y(m)': fmt3(y_e),
            '烟幕干扰弹起爆点z(m)': fmt3(z_e),
            '投放时刻D(s)': fmt3(D),
            '起爆时刻E(s)': fmt3(E),
            '引信延时Δ=E-D(s)': fmt3(delta),
            '单弹有效干扰时长(s)': fmt3(tot_i),
        })

    df = pd.DataFrame(rows, columns=[
        '无人机编号','无人机运动方向(度)','无人机运动速度(m/s)','烟幕干扰弹编号',
        '烟幕干扰弹投放点x(m)','烟幕干扰弹投放点y(m)','烟幕干扰弹投放点z(m)',
        '烟幕干扰弹起爆点x(m)','烟幕干扰弹起爆点y(m)','烟幕干扰弹起爆点z(m)',
        '投放时刻D(s)','起爆时刻E(s)','引信延时Δ=E-D(s)','单弹有效干扰时长(s)'
    ])

    # 保存详细结果到output目录
    out_path_detail = "./output/q4_detailed_results.xlsx"
    with pd.ExcelWriter(out_path_detail, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Sheet1")
    print(f"✓ 详细结果已保存到: {out_path_detail}")

    # 按照题目要求的格式保存Q4结果 - Q4对应的是q4_result2_data.xlsx
    df_template = pd.DataFrame([
        {
            '无人机编号': df.loc[i, '无人机编号'],
            '无人机运动方向(度)': df.loc[i, '无人机运动方向(度)'],
            '无人机运动速度(m/s)': df.loc[i, '无人机运动速度(m/s)'],
            '烟幕干扰弹投放点的x坐标(m)': df.loc[i, '烟幕干扰弹投放点x(m)'],
            '烟幕干扰弹投放点的y坐标(m)': df.loc[i, '烟幕干扰弹投放点y(m)'],
            '烟幕干扰弹投放点的z坐标(m)': df.loc[i, '烟幕干扰弹投放点z(m)'],
            '烟幕干扰弹起爆点的x坐标(m)': df.loc[i, '烟幕干扰弹起爆点x(m)'],
            '烟幕干扰弹起爆点的y坐标(m)': df.loc[i, '烟幕干扰弹起爆点y(m)'],
            '烟幕干扰弹起爆点的z坐标(m)': df.loc[i, '烟幕干扰弹起爆点z(m)'],
            '有效干扰时长(s)': df.loc[i, '单弹有效干扰时长(s)'],
        }
        for i in range(len(df))
    ])

    out_path_template = "./output/q4_result2_data.xlsx"
    with pd.ExcelWriter(out_path_template, engine="openpyxl") as w:
        df_template.to_excel(w, index=False, sheet_name="Sheet1")
    print(f"✓ 结果已保存到: {out_path_template}")
    
    print("\n📁 生成的文件:")
    print("  - output/q4_result2_data.xlsx (Q4标准结果，对应题目result2.xlsx)")
    print("  - output/q4_detailed_results.xlsx (详细分析数据)")
    print("  - output/q4_3d_scenario.png (三维场景图)")
    print("  - output/q4_top_view.png (俯视图)")
    print("  - output/q4_optimization_analysis.png (优化分析图)")
    
    print(f"\n✅ Q4问题求解完成！联合遮蔽时长: {U:.3f} 秒")


if __name__ == "__main__":
    main()
