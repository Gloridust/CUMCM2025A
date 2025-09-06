#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
q1_solver.py
第一问固定场景：FY1 以 120 m/s 朝向假目标方向飞行，
受领任务 1.5 s 后投放，间隔 3.6 s 后起爆。
使用 LOS 阻断（线段与球体交叉）判定，圆柱体采样。
输出：result_q1.xlsx（包含投放点、起爆点、总遮蔽时间、遮蔽区间）
"""
import numpy as np, math, pandas as pd, os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 常量（题目给定） ----------------
g = 9.8
R_smoke = 10.0
sink = 3.0
shield_T = 20.0

v_missile = 300.0
M0 = np.array([20000.0, 0.0, 2000.0])      # 导弹初始
U0 = np.array([17800.0, 0.0, 1800.0])      # FY1 初始
v_uav = 120.0
t_drop = 1.5        # 受领任务 1.5s 后投放
delay_after_drop = 3.6
t_explode = t_drop + delay_after_drop

# target cylinder
x_c, y_c = 0.0, 200.0
r_true = 7.0
z0_true = 0.0
h_true = 10.0

# missile unit dir to fake target (origin)
uM = -M0 / np.linalg.norm(M0)
t_hit = np.linalg.norm(M0 - np.array([0.0,0.0,0.0])) / v_missile

# ---------------- 辅助函数 ----------------
def missile_pos(t):
    return M0 + v_missile * uM * t

def uav_unit_dir_to_origin(U0=U0):
    horiz = np.array([-U0[0], -U0[1], 0.0])
    return horiz / np.linalg.norm(horiz)

Udir = uav_unit_dir_to_origin()

def explosion_center():
    # 投放时无人机仍等速直线飞行
    r_drop = U0 + v_uav * Udir * t_drop
    dt = t_explode - t_drop
    r_expl = r_drop + v_uav * Udir * dt + np.array([0.0,0.0,-0.5*g*dt*dt])
    return r_expl

R_expl = explosion_center()

def cloud_center(t):
    # 当 t < t_explode 应该无效，但函数仅在 t>=t_explode 时调用
    return R_expl + np.array([0.0,0.0,-sink*(t - t_explode)])

def cylinder_samples(n_th=36, n_z=5, face_rs=(0.0, 3.5, None)):
    if face_rs[-1] is None:
        face_rs = list(face_rs[:-1]) + [r_true]
    pts=[]
    thetas = np.linspace(0,2*math.pi, n_th, endpoint=False)
    zs = np.linspace(z0_true, z0_true+h_true, n_z)
    for z in zs:
        for th in thetas:
            x = x_c + r_true * math.cos(th)
            y = y_c + r_true * math.sin(th)
            pts.append((x,y,z))
    for z_fixed in (z0_true, z0_true+h_true):
        for r in face_rs:
            for th in thetas:
                x = x_c + r * math.cos(th)
                y = y_c + r * math.sin(th)
                pts.append((x,y,z_fixed))
    return np.array(pts, dtype=float)

SAMPLES = cylinder_samples(n_th=48, n_z=6, face_rs=(0.0, r_true/2, None))

def point_to_segment_dist(P,A,B):
    AB = B - A
    denom = float(np.dot(AB,AB))
    if denom == 0.0:
        return float(np.linalg.norm(P - A))
    tt = float(np.dot(P - A, AB) / denom)
    tt = 0.0 if tt < 0.0 else (1.0 if tt > 1.0 else tt)
    return float(np.linalg.norm(P - (A + tt*AB)))

def make_g_cylinder_all(samples):
    samples = np.asarray(samples, dtype=float)
    def g(t):
        P = cloud_center(t)
        A = missile_pos(t)
        dmax = 0.0
        for Q in samples:
            d = point_to_segment_dist(P, A, Q)
            if d > dmax: dmax = d
        return dmax - R_smoke
    return g

def intervals_from_g(gfunc, t0, t1, dt_scan=0.01, tol=1e-4):
    times=[t0]; cur=t0
    while cur < t1:
        cur = min(cur + dt_scan, t1); times.append(cur)
    vals = [gfunc(tt) for tt in times]
    def bisect(a,b,fa,fb):
        if fa==0.0: return a
        if fb==0.0: return b
        lo,hi,flo,fhi = a,b,fa,fb
        for _ in range(64):
            mid = 0.5*(lo+hi); fm = gfunc(mid)
            if abs(fm) < 1e-12 or (hi-lo) < tol:
                return mid
            if flo*fm <= 0.0:
                hi,fhi = mid,fm
            else:
                lo,flo = mid,fm
        return 0.5*(lo+hi)
    roots=[]
    for i in range(1,len(times)):
        a,b = times[i-1], times[i]; fa,fb = vals[i-1], vals[i]
        if fa==0.0: roots.append(a)
        if fa*fb < 0.0:
            roots.append(bisect(a,b,fa,fb))
    roots = sorted(set(roots))
    intervals=[]
    inside = (vals[0] <= 0.0)
    start = t0 if inside else None
    for r in roots:
        if inside:
            intervals.append((start, r)); inside=False; start=None
        else:
            inside=True; start=r
    if inside: intervals.append((start, t1))
    total = sum(b-a for a,b in intervals)
    return total, intervals

# ---------------- 可视化函数 ----------------
def create_visualizations(intervals, total):
    """创建Q1问题的可视化图表"""
    os.makedirs("./output", exist_ok=True)
    
    # 1. 三维场景图
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
    
    # 绘制烟幕球体（在起爆时刻）
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
    
    # 2. 俯视图（XY平面）
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
    
    # 3. 遮蔽时间分析图
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

# ---------------- 主流程 ----------------
def main():
    print("=" * 60)
    print("Q1: 单无人机单烟幕弹对M1导弹干扰问题求解")
    print("=" * 60)
    
    # 确保output目录存在
    os.makedirs("./output", exist_ok=True)
    
    t0 = max(t_explode, 0.0)
    t1 = min(t_explode + shield_T, t_hit)
    g_all = make_g_cylinder_all(SAMPLES)
    total, intervals = intervals_from_g(g_all, t0, t1, dt_scan=0.01, tol=1e-4)
    
    print("\n📊 计算结果:")
    print(f"  起爆时间: {t_explode:.2f} 秒")
    print(f"  起爆中心: ({R_expl[0]:.1f}, {R_expl[1]:.1f}, {R_expl[2]:.1f}) 米")
    print(f"  导弹撞击时间: {t_hit:.2f} 秒")
    print(f"  总遮蔽时间: {total:.6f} 秒")
    print(f"  遮蔽区间: {[(round(a,3), round(b,3)) for a,b in intervals]}")
    
    # 诊断最难遮住点
    if intervals:
        tmid = 0.5*(intervals[0][0] + intervals[0][1])
        P = cloud_center(tmid); A = missile_pos(tmid)
        dists = [point_to_segment_dist(P, A, Q) for Q in SAMPLES]
        idx = int(np.argmax(dists))
        print(f"  最难遮蔽点: {tuple(np.round(SAMPLES[idx],1))} 米")
        print(f"  距离-半径: {round(dists[idx]-R_smoke,3)} 米")
    
    # 生成可视化图表
    print("\n🎨 生成可视化图表...")
    create_visualizations(intervals, total)
    
    # 保存Excel文件到output目录
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
    
    # 同时保存一个详细版本用于调试
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
    
    print("\n📁 生成的文件:")
    print("  - output/q1_data.xlsx (Q1标准结果)")
    print("  - output/q1_detailed_results.xlsx (详细分析数据)")
    print("  - output/q1_3d_scenario.png (三维场景图)")
    print("  - output/q1_top_view.png (俯视图)")
    print("  - output/q1_time_analysis.png (时间分析图)")
    print("\n✅ Q1问题求解完成！")

if __name__ == "__main__":
    main()
