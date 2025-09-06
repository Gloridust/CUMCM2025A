#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q3_solver.py
Q3: FY1 投放 3 枚烟幕干扰弹，最大化对 M1 的遮蔽时间（LOS 判定）。
实现思路：粗扫(v,theta) -> 为每个投放时刻生成候选 E -> 组合并集 -> 局部精化 E 与 v,theta -> 输出 result1.xlsx
注意：FY1 投三枚，这里按 Q3 要求执行。
"""
import numpy as np, math, pandas as pd, os, time
from math import cos, sin, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 基本常量 ----------------
g = 9.8
R = 10.0
sink = 3.0
shield_T = 20.0
explode_delay = 3.6   # friend code used 3.6s in Q1; we keep 3.6 if consistent. (可参数化)

# UAV/Missile/Target
T = np.array([0.0, 200.0, 0.0])
M0 = np.array([20000.0, 0.0, 2000.0])
U0 = np.array([17800.0, 0.0, 1800.0])
v_missile = 300.0
U_MISSILE = -M0 / np.linalg.norm(M0)

# 三次投放假定时刻（起始投放时间位置），这里用 0,1,2 s 作为 drop times
DROP_TIMES = [0.0, 1.0, 2.0]

# ------------- 事件驱动遮蔽判定（同朋友代码风格） -------------
def missile_pos(t):
    return M0 + v_missile * U_MISSILE * t

def explosion_point(v_uav, theta, t_drop, explode_delay_local=explode_delay):
    r_drop = U0 + v_uav * np.array([cos(theta), sin(theta), 0.0]) * t_drop
    dt = explode_delay_local
    # horizontal continues with UAV speed, vertical free fall
    return r_drop + v_uav * np.array([cos(theta), sin(theta), 0.0]) * dt + np.array([0.0, 0.0, -0.5*g*dt*dt])

def cloud_center_at(t, r_expl, t_expl):
    return r_expl + np.array([0.0,0.0,-sink*(t - t_expl)])

def point_to_segment_dist(P, A, B):
    AB = B - A; denom = float(np.dot(AB,AB))
    if denom == 0.0: return float(np.linalg.norm(P-A))
    tt = float(np.dot(P-A,AB)/denom); tt = max(0.0, min(1.0, tt))
    return float(np.linalg.norm(P - (A + tt*AB)))

def intervals_for_explosion(t_expl, x_e, y_e, z_e, dt_scan=0.02, tol=1e-4):
    t0, t1 = t_expl, t_expl + shield_T
    if t1 <= t0: return 0.0, []
    def center(t): return np.array([x_e, y_e, z_e - sink*(t - t_expl)], dtype=float)
    def g(t): return point_to_segment_dist(center(t), missile_pos(t), T) - R
    # scan grid and find sign changes
    ts = [t0]; cur=t0
    while cur < t1:
        cur = min(cur + dt_scan, t1); ts.append(cur)
    vs = [g(t) for t in ts]
    # if everywhere positive and large, skip
    if min(vs) > 8.0: return 0.0, []
    # bisection helper
    def root(a,b,fa,fb):
        if fa == 0.0: return a
        if fb == 0.0: return b
        lo,hi,flo,fhi = a,b,fa,fb
        for _ in range(64):
            mid = 0.5*(lo+hi); fm = g(mid)
            if abs(fm) < 1e-12 or (hi-lo) < tol: return mid
            if flo*fm <= 0.0: hi,fhi = mid,fm
            else: lo,flo = mid,fm
        return 0.5*(lo+hi)
    roots=[]
    for i in range(1,len(ts)):
        a,b = ts[i-1], ts[i]; fa, fb = vs[i-1], vs[i]
        if fa == 0.0: roots.append(a)
        if fa*fb < 0.0: roots.append(root(a,b,fa,fb))
    roots = sorted(set(roots))
    intervals=[]; inside = (vs[0] <= 0.0); start = t0 if inside else None
    for r in roots:
        if inside:
            intervals.append((start, r)); inside=False; start=None
        else:
            inside=True; start=r
    if inside: intervals.append((start, t1))
    return sum(b-a for a,b in intervals), intervals

def merge_intervals(intervals):
    if not intervals: return 0.0, []
    xs = sorted(intervals, key=lambda x: x[0])
    merged = [list(xs[0])]
    for a,b in xs[1:]:
        if a <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], b)
        else: merged.append([a,b])
    merged = [(a,b) for a,b in merged]
    return sum(b-a for a,b in merged), merged

# ---------------- 单弹候选生成与三弹组合 ----------------
def single_candidates_for_drop(D_i, v, theta_deg, tstep=0.2, dt_scan=0.02, topk=24):
    th = math.radians(theta_deg); ux,uy = math.cos(th), math.sin(th)
    texps = np.round(np.arange(max(D_i,0.0), 20.0 + 1e-9, tstep), 4)
    out=[]
    for E in texps:
        # explosion point computed from E and D_i (E is explosion time)
        # find drop point r_drop at drop time D_i: r_drop = U0 + v*dir*D_i
        delay = E - D_i
        # explosion coordinates: horizontal moved for delay seconds with UAV speed, vertical free fall for delay
        x_e = U0[0] + v*ux*E
        y_e = U0[1] + v*uy*E
        z_e = 1800.0 - 0.5*g*(delay**2)
        if z_e < 0: continue
        tot, ints = intervals_for_explosion(E, x_e, y_e, z_e, dt_scan=dt_scan)
        if tot > 0:
            out.append((float(E), float(tot), ints))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:min(topk, len(out))]

def search_three_charges(v, theta_deg, tstep, dt_scan, topk):
    all_cands = []
    for i, Di in enumerate(DROP_TIMES):
        lst = single_candidates_for_drop(Di, v, theta_deg, tstep=tstep, dt_scan=dt_scan, topk=topk)
        if not lst: return None
        all_cands.append(lst)
    best=None
    for e1 in all_cands[0]:
        for e2 in all_cands[1]:
            for e3 in all_cands[2]:
                ints_sum = e1[2] + e2[2] + e3[2]
                union_total, union_ints = merge_intervals(ints_sum)
                if (best is None) or (union_total > best[0]):
                    best = (union_total, (e1[0], e2[0], e3[0]), union_ints)
    return best

# ---------------- 精化 E（坐标下降） ----------------
def refine_E(v, theta_deg, E_tuple, dt_scan, step0=0.4, step_min=0.02, iters=60):
    Es = list(E_tuple)
    th = math.radians(theta_deg); ux,uy = math.cos(th), math.sin(th)
    def eval_union(Es_cur):
        ints_all=[]
        for i, E in enumerate(Es_cur):
            Ei = max(E, DROP_TIMES[i])
            x_e = U0[0] + v*ux*Ei
            y_e = U0[1] + v*uy*Ei
            z_e = 1800.0 - 0.5*g*(Ei - DROP_TIMES[i])**2
            if z_e < 0: return -1.0, []
            tot, ints = intervals_for_explosion(Ei, x_e, y_e, z_e, dt_scan=dt_scan)
            if tot <= 0: return 0.0, []
            ints_all += ints
        return merge_intervals(ints_all)
    best_val, best_ints = eval_union(Es)
    step = step0
    for _ in range(iters):
        improved=False
        for idx in range(3):
            for sgn in (+1,-1):
                trial = Es[:]
                trial[idx] = max(DROP_TIMES[idx], trial[idx] + sgn*step)
                val, uni = eval_union(trial)
                if val > best_val + 1e-6:
                    Es, best_val, best_ints = trial, val, uni
                    improved=True; break
            if improved: continue
        if not improved:
            step *= 0.5
            if step < step_min: break
    return best_val, tuple(Es), best_ints

# ---------------- v,theta 细化（交替坐标下降） ----------------
def refine_v_theta(v0, theta0_deg, E0, cfg):
    v = float(np.clip(v0, 70.0, 140.0))
    th_deg = float(theta0_deg)
    Es = tuple(E0)
    best_val, Es, uni = refine_E(v, th_deg, Es, dt_scan=cfg['DT_FINE'],
                                 step0=cfg.get('REF_E_STEP0',0.4), step_min=cfg.get('REF_E_MIN',0.02), iters=cfg.get('REF_E_ITERS',60))
    step_v = cfg.get('REF_V0H',0.2); step_th = cfg.get('REF_TH0',0.1)
    for _ in range(cfg.get('REF_VTH_ITERS',35)):
        improved=False
        for var in ('v','th'):
            for sgn in (+1,-1):
                vt, tht = v, th_deg
                if var=='v':
                    vt = float(np.clip(v + sgn*step_v, 70.0, 140.0))
                else:
                    tht = th_deg + sgn*step_th
                ans = search_three_charges(vt, tht, tstep=cfg['TEXP_STEP_FINE'], dt_scan=cfg['DT_FINE'], topk=min(24,cfg['TOPK']))
                if ans is None: continue
                _, E_init, _ = ans
                val, Es_new, uni_new = refine_E(vt, tht, E_init, dt_scan=cfg['DT_FINE'], step0=cfg.get('REF_E_STEP0',0.4), step_min=cfg.get('REF_E_MIN',0.02), iters=cfg.get('REF_E_ITERS',60))
                if val > best_val + 1e-6:
                    v, th_deg, Es, best_val, uni = vt, tht, Es_new, val, uni_new
                    improved=True; break
            if improved: break
        if not improved:
            step_v *= 0.5; step_th *= 0.5
            if step_v < cfg.get('REF_V_MIN',0.01) and step_th < cfg.get('REF_TH_MIN',0.01):
                break
    return best_val, v, th_deg, Es, uni

# ---------------- 可视化模块 ----------------
def create_visualizations(v_best, th_best, E_best, uni_best, tot):
    """创建Q3问题的可视化图表"""
    os.makedirs("./output", exist_ok=True)
    
    th_rad = math.radians(th_best)
    ux, uy = math.cos(th_rad), math.sin(th_rad)
    
    # 计算关键位置
    explosion_points = []
    drop_points = []
    for k, E in enumerate(E_best):
        Dk = DROP_TIMES[k]
        x_e = U0[0] + v_best * ux * E
        y_e = U0[1] + v_best * uy * E
        z_e = 1800.0 - 0.5 * g * (E - Dk)**2
        explosion_points.append([x_e, y_e, z_e])
        
        r_drop = U0 + v_best * np.array([ux, uy, 0.0]) * Dk
        drop_points.append(r_drop)
    
    explosion_points = np.array(explosion_points)
    drop_points = np.array(drop_points)
    
    # 1. 三维场景图
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制导弹轨迹
    missile_flight_time = np.linalg.norm(M0) / v_missile
    t_trajectory = np.linspace(0, missile_flight_time, 100)
    missile_trajectory = np.array([missile_pos(t) for t in t_trajectory])
    ax.plot(missile_trajectory[:, 0], missile_trajectory[:, 1], missile_trajectory[:, 2], 
            'r-', linewidth=3, label='导弹M1轨迹')
    
    # 绘制无人机轨迹
    t_uav = np.linspace(0, max(E_best) * 1.2, 100)
    uav_trajectory = np.array([U0 + v_best * np.array([ux, uy, 0.0]) * t for t in t_uav])
    ax.plot(uav_trajectory[:, 0], uav_trajectory[:, 1], uav_trajectory[:, 2], 
            'b-', linewidth=2, label='无人机FY1轨迹')
    
    # 标记关键点
    ax.scatter(*M0, color='red', s=120, label='导弹初始位置M1')
    ax.scatter(*U0, color='blue', s=120, label='无人机初始位置FY1')
    ax.scatter(0, 0, 0, color='black', s=120, marker='s', label='假目标')
    ax.scatter(*T, color='green', s=120, marker='^', label='真目标')
    
    # 绘制3个投放点和起爆点
    colors = ['orange', 'cyan', 'magenta']
    for i, (drop_pt, expl_pt, color) in enumerate(zip(drop_points, explosion_points, colors)):
        ax.scatter(*drop_pt, color=color, s=180, marker='*', 
                  label=f'烟幕弹{i+1}投放点')
        ax.scatter(*expl_pt, color=color, s=180, marker='o', 
                  label=f'烟幕弹{i+1}起爆点')
        
        # 绘制烟幕球体
        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        x_sphere = R * np.outer(np.cos(u), np.sin(v)) + expl_pt[0]
        y_sphere = R * np.outer(np.sin(u), np.sin(v)) + expl_pt[1]
        z_sphere = R * np.outer(np.ones(np.size(u)), np.cos(v)) + expl_pt[2]
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
    ax.set_title('Q3: 单无人机三烟幕弹协同干扰三维场景图', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('./output/q3_3d_scenario.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 俯视图（XY平面）
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 绘制轨迹投影
    ax.plot(missile_trajectory[:, 0], missile_trajectory[:, 1], 'r-', linewidth=3, label='导弹M1轨迹')
    ax.plot(uav_trajectory[:, 0], uav_trajectory[:, 1], 'b-', linewidth=2, label='无人机FY1轨迹')
    
    # 标记关键点
    ax.scatter(M0[0], M0[1], color='red', s=120, label='导弹初始位置M1')
    ax.scatter(U0[0], U0[1], color='blue', s=120, label='无人机初始位置FY1')
    ax.scatter(0, 0, color='black', s=120, marker='s', label='假目标')
    ax.scatter(T[0], T[1], color='green', s=120, marker='^', label='真目标')
    
    # 绘制3个投放点、起爆点和烟幕覆盖区域
    for i, (drop_pt, expl_pt, color) in enumerate(zip(drop_points, explosion_points, colors)):
        ax.scatter(drop_pt[0], drop_pt[1], color=color, s=180, marker='*', 
                  label=f'烟幕弹{i+1}投放点')
        ax.scatter(expl_pt[0], expl_pt[1], color=color, s=180, marker='o', 
                  label=f'烟幕弹{i+1}起爆点')
        
        # 绘制烟幕覆盖区域
        smoke_circle = patches.Circle((expl_pt[0], expl_pt[1]), R, linewidth=2, 
                                    edgecolor=color, facecolor=color, alpha=0.2, 
                                    label=f'烟幕{i+1}覆盖区域' if i < 3 else "")
        ax.add_patch(smoke_circle)
    
    # 绘制真目标圆柱体俯视图
    circle_true = patches.Circle((T[0], T[1]), 7, linewidth=2, edgecolor='green', 
                               facecolor='lightgreen', alpha=0.3, label='真目标保护区')
    ax.add_patch(circle_true)
    
    # 绘制遮蔽时间段的视线
    if uni_best:
        for i, (start, end) in enumerate(uni_best[:3]):  # 最多显示3个时段
            mid_time = (start + end) / 2
            missile_pos_mid = missile_pos(mid_time)
            ax.plot([missile_pos_mid[0], T[0]], [missile_pos_mid[1], T[1]], 
                   '--', linewidth=2, alpha=0.6, 
                   label=f'遮蔽时段{i+1}视线' if i < 3 else "")
    
    ax.set_xlabel('X (米)')
    ax.set_ylabel('Y (米)')
    ax.set_title('Q3: 单无人机三烟幕弹协同干扰俯视图', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('./output/q3_top_view.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 时间轴分析图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # 上图：每个烟幕弹的单独遮蔽效果
    for i, E in enumerate(E_best):
        Dk = DROP_TIMES[i]
        x_e = U0[0] + v_best * ux * E
        y_e = U0[1] + v_best * uy * E
        z_e = 1800.0 - 0.5 * g * (E - Dk)**2
        
        single_total, single_intervals = intervals_for_explosion(E, x_e, y_e, z_e, dt_scan=0.02)
        
        # 绘制单个烟幕弹的遮蔽时间段
        for start, end in single_intervals:
            ax1.barh(i, end-start, left=start, height=0.6, 
                    color=colors[i], alpha=0.7, 
                    label=f'烟幕弹{i+1}' if start == single_intervals[0][0] else "")
        
        # 标记起爆时间
        ax1.axvline(E, color=colors[i], linestyle=':', alpha=0.8)
        ax1.text(E, i+0.3, f'起爆{i+1}', rotation=90, fontsize=8, ha='center')
    
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('烟幕弹编号')
    ax1.set_yticks(range(3))
    ax1.set_yticklabels(['烟幕弹1', '烟幕弹2', '烟幕弹3'])
    ax1.set_title('各烟幕弹单独遮蔽效果', fontweight='bold')
    ax1.legend()
    ax1.grid(True, axis='x')
    
    # 下图：合并后的总遮蔽效果
    ax2.barh(0, missile_flight_time, height=0.3, color='lightcoral', alpha=0.7, label='导弹飞行时间')
    
    # 绘制合并后的遮蔽区间
    for i, (start, end) in enumerate(uni_best):
        ax2.barh(0.5, end-start, left=start, height=0.3, color='green', alpha=0.8, 
                label='协同遮蔽时间' if i == 0 else "")
    
    ax2.set_xlabel('时间 (秒)')
    ax2.set_yticks([0, 0.5])
    ax2.set_yticklabels(['导弹飞行', '协同遮蔽'])
    ax2.set_title(f'协同遮蔽总效果: {tot:.3f}秒', fontweight='bold')
    ax2.legend()
    ax2.grid(True, axis='x')
    
    plt.tight_layout()
    plt.savefig('./output/q3_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 优化过程分析图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 遮蔽时间随速度变化
    speeds = np.linspace(70, 140, 30)
    speed_coverage = []
    for v_test in speeds:
        # 使用最优角度和起爆时间测试不同速度
        test_result = search_three_charges(v_test, th_best, tstep=0.3, dt_scan=0.1, topk=8)
        if test_result:
            speed_coverage.append(test_result[0])
        else:
            speed_coverage.append(0)
    
    ax1.plot(speeds, speed_coverage, 'b-', linewidth=2)
    ax1.axvline(v_best, color='r', linestyle='--', label=f'最优速度: {v_best:.1f}m/s')
    ax1.set_xlabel('无人机速度 (m/s)')
    ax1.set_ylabel('总遮蔽时间 (秒)')
    ax1.set_title('遮蔽时间随无人机速度变化', fontweight='bold')
    ax1.legend()
    ax1.grid(True)
    
    # 子图2: 遮蔽时间随角度变化
    angles = np.linspace(th_best-2, th_best+2, 30)
    angle_coverage = []
    for th_test in angles:
        test_result = search_three_charges(v_best, th_test, tstep=0.3, dt_scan=0.1, topk=8)
        if test_result:
            angle_coverage.append(test_result[0])
        else:
            angle_coverage.append(0)
    
    ax2.plot(angles, angle_coverage, 'g-', linewidth=2)
    ax2.axvline(th_best, color='r', linestyle='--', label=f'最优角度: {th_best:.1f}°')
    ax2.set_xlabel('飞行角度 (度)')
    ax2.set_ylabel('总遮蔽时间 (秒)')
    ax2.set_title('遮蔽时间随飞行角度变化', fontweight='bold')
    ax2.legend()
    ax2.grid(True)
    
    # 子图3: 起爆时间序列
    ax3.bar(range(1, 4), E_best, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('烟幕弹编号')
    ax3.set_ylabel('起爆时间 (秒)')
    ax3.set_title('各烟幕弹起爆时间序列', fontweight='bold')
    ax3.set_xticks(range(1, 4))
    ax3.grid(True, axis='y')
    
    # 在柱状图上添加数值标签
    for i, e_time in enumerate(E_best):
        ax3.text(i+1, e_time + 0.1, f'{e_time:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    # 子图4: 投放时间vs起爆时间
    drop_times = DROP_TIMES
    explode_times = E_best
    delays = [e - d for e, d in zip(explode_times, drop_times)]
    
    ax4.scatter(drop_times, explode_times, c=colors, s=150, alpha=0.7, edgecolor='black')
    for i, (dt, et, delay) in enumerate(zip(drop_times, explode_times, delays)):
        ax4.annotate(f'弹{i+1}\n延迟{delay:.2f}s', (dt, et), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    # 绘制对角线参考
    min_time = min(min(drop_times), min(explode_times))
    max_time = max(max(drop_times), max(explode_times))
    ax4.plot([min_time, max_time], [min_time, max_time], 'k--', alpha=0.5, label='无延迟线')
    
    ax4.set_xlabel('投放时间 (秒)')
    ax4.set_ylabel('起爆时间 (秒)')
    ax4.set_title('投放时间 vs 起爆时间关系', fontweight='bold')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('./output/q3_optimization_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Q3可视化图表已保存到output/目录")

def save_results_to_excel(v_best, th_best, E_best, uni_best, tot):
    """保存结果到Excel文件"""
    os.makedirs("./output", exist_ok=True)
    
    th_rad = math.radians(th_best)
    ux, uy = math.cos(th_rad), math.sin(th_rad)
    heading_deg = int(round((math.degrees(math.atan2(uy, ux))) % 360))
    
    # 按照题目要求的格式保存Q3结果
    rows = []
    for k, E in enumerate(E_best, start=1):
        Dk = DROP_TIMES[k-1]
        x_e = U0[0] + v_best * ux * E
        y_e = U0[1] + v_best * uy * E
        z_e = 1800.0 - 0.5 * g * (E - Dk)**2
        r_drop = U0 + v_best * np.array([ux, uy, 0.0]) * Dk
        ti, _ = intervals_for_explosion(E, x_e, y_e, z_e, dt_scan=0.06)
        
        rows.append({
            '无人机运动方向': heading_deg,
            '无人机运动速度 (m/s)': round(v_best, 1),
            '烟幕干扰弹编号（1 2 3）': k,
            '烟幕干扰弹投放点的x坐标 (m)': round(r_drop[0], 1),
            '烟幕干扰弹投放点的y坐标 (m)': round(r_drop[1], 1),
            '烟幕干扰弹投放点的z坐标 (m)': round(r_drop[2], 1),
            '烟幕干扰弹起爆点的x坐标 (m)': round(x_e, 1),
            '烟幕干扰弹起爆点的y坐标 (m)': round(y_e, 1),
            '烟幕干扰弹起爆点的z坐标 (m)': round(z_e, 1),
            '有效干扰时长 (s)': round(ti, 3)
        })
    
    df = pd.DataFrame(rows)
    
    # 保存到output目录 - Q3对应的是q3_result1_data.xlsx
    output_file = "./output/q3_result1_data.xlsx"
    df.to_excel(output_file, index=False)
    print(f"✓ 结果已保存到: {output_file}")
    
    # 保存详细分析结果
    detailed_rows = [{
        'scenario': 'Q3_three_bombs',
        'uav_id': 'FY1',
        'optimization_method': 'coordinate_descent',
        'uav_direction_deg': heading_deg,
        'uav_speed_m_s': round(v_best, 1),
        'flight_angle_rad': round(th_rad, 4),
        'num_bombs': 3,
        'drop_times': str([round(dt, 3) for dt in DROP_TIMES]),
        'explode_times': str([round(e, 3) for e in E_best]),
        'explode_delays': str([round(e-d, 3) for e, d in zip(E_best, DROP_TIMES)]),
        'total_coverage_time_s': round(tot, 6),
        'coverage_intervals': str([(round(a, 3), round(b, 3)) for a, b in uni_best]),
        'num_intervals': len(uni_best),
        'individual_coverage_times': str([round(intervals_for_explosion(
            E_best[i], 
            U0[0] + v_best * ux * E_best[i],
            U0[1] + v_best * uy * E_best[i],
            1800.0 - 0.5 * g * (E_best[i] - DROP_TIMES[i])**2,
            dt_scan=0.06)[0], 3) for i in range(3)])
    }]
    
    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_excel("./output/q3_detailed_results.xlsx", index=False)
    
    print("\n📁 生成的文件:")
    print("  - output/q3_result1_data.xlsx (Q3标准结果，对应题目result1.xlsx)")
    print("  - output/q3_detailed_results.xlsx (详细分析数据)")
    print("  - output/q3_3d_scenario.png (三维场景图)")
    print("  - output/q3_top_view.png (俯视图)")
    print("  - output/q3_time_analysis.png (时间轴分析图)")
    print("  - output/q3_optimization_analysis.png (优化分析图)")

# ---------------- 主流程（粗扫→精化） ----------------
def main():
    # 使用 balanced 模式的参数（借鉴你朋友代码）
    CFG = {
        'V_RANGE': (138.8, 140.2, 0.05),
        'TH_RANGE': (179.0, 0.8, 0.05),
        'TEXP_STEP_COARSE': 0.20, 'DT_COARSE': 0.12, 'TOPK': 32,
        'N_TOP_COMBOS': 16,
        'TEXP_STEP_FINE': 0.05, 'DT_FINE': 0.06,
        'REF_E_STEP0':0.40, 'REF_E_MIN':0.02, 'REF_E_ITERS':60,
        'REF_V0H':0.20,'REF_V_MIN':0.01,'REF_TH0':0.10,'REF_TH_MIN':0.01,
        'REF_VTH_ITERS':35
    }
    cfg = CFG
    vmin, vmax, vstep = cfg['V_RANGE']
    th_c, th_span, th_step = cfg['TH_RANGE']
    # build v grid clipped to [70,140]
    vs = []
    val = vmin
    while val <= vmax + 1e-9:
        vs.append(round(min(140.0, max(70.0, val)),6)); val += vstep
    ths = []
    tstart = th_c - th_span; tend = th_c + th_span
    t = tstart
    while t <= tend + 1e-9:
        ths.append(round(t, 6)); t += th_step

    combos=[]
    total = len(vs)*len(ths); done=0
    print("Coarse sweep total tasks:", total)
    for v in vs:
        for th in ths:
            ans = search_three_charges(v, th, tstep=cfg['TEXP_STEP_COARSE'], dt_scan=cfg['DT_COARSE'], topk=cfg['TOPK'])
            done += 1
            if done % max(1, total//10) == 0:
                cur_best = max([c[0] for c in combos], default=0.0)
                print(f"[coarse] {done}/{total} done, current best {cur_best:.3f}s")
            if ans is None: continue
            combos.append((ans[0], v, th, ans[1], ans[2]))
    if not combos:
        print("No feasible combos found in coarse sweep. Try wider ranges.")
        return
    combos.sort(key=lambda x: x[0], reverse=True)
    topN = combos[:cfg['N_TOP_COMBOS']]
    print("Top coarse candidates:", [round(x[0],3) for x in topN])

    best_final=None
    for idx, (u_tot, v, th, E0, _) in enumerate(topN, start=1):
        print(f"[refine] candidate {idx}/{len(topN)} starting v={v}, th={th}")
        val, v2, th2, E2, uni2 = refine_v_theta(v, th, E0, cfg)
        print(f"  local refined = {val:.3f}s")
        if best_final is None or val > best_final[0]:
            best_final = (val, v2, th2, E2, uni2)

    if best_final is None:
        print("Refinement failed to find improved solution.")
        return
    
    tot, v_best, th_best, E_best, uni_best = best_final
    
    print("=" * 60)
    print("Q3: 单无人机三烟幕弹协同干扰问题求解完成")
    print("=" * 60)
    
    print(f"\n📊 最优解结果:")
    print(f"  总遮蔽时间: {tot:.3f} 秒")
    print(f"  最优速度: {v_best:.1f} m/s")
    print(f"  最优角度: {th_best:.1f}°")
    print(f"  起爆时间序列: {[round(e, 2) for e in E_best]} 秒")
    print(f"  遮蔽区间数量: {len(uni_best)} 个")
    print(f"  遮蔽区间: {[(round(a, 2), round(b, 2)) for a, b in uni_best]}")
    
    # 生成可视化图表
    print("\n🎨 生成可视化图表...")
    create_visualizations(v_best, th_best, E_best, uni_best, tot)
    
    # 保存结果到Excel
    print("\n💾 保存结果到Excel...")
    save_results_to_excel(v_best, th_best, E_best, uni_best, tot)
    
    print(f"\n✅ Q3问题求解完成！总遮蔽时间: {tot:.3f} 秒")

if __name__ == '__main__':
    main()
