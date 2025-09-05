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

# ---------------- 主流程 ----------------
def main():
    t0 = max(t_explode, 0.0)
    t1 = min(t_explode + shield_T, t_hit)
    g_all = make_g_cylinder_all(SAMPLES)
    total, intervals = intervals_from_g(g_all, t0, t1, dt_scan=0.01, tol=1e-4)
    print("Q1 fixed scenario results:")
    print("Explosion time t_explode =", t_explode, "s")
    print("Explosion center:", tuple(R_expl))
    print("Missile hit time (fake) t_hit =", t_hit, "s")
    print("Total covered_time:", round(total,6), "s")
    print("Intervals:", [(round(a,6), round(b,6)) for a,b in intervals])
    # 诊断最难遮住点（在第一个 interval 中点）
    if intervals:
        tmid = 0.5*(intervals[0][0] + intervals[0][1])
        P = cloud_center(tmid); A = missile_pos(tmid)
        dists = [point_to_segment_dist(P, A, Q) for Q in SAMPLES]
        idx = int(np.argmax(dists))
        print("Hardest sample point near:", tuple(np.round(SAMPLES[idx],3)), "dist-R =", round(dists[idx]-R_smoke,6))
    # 保存 Excel
    rows = [{
        'uav_direction_deg': (math.degrees(math.atan2(Udir[1], Udir[0])) % 360),
        'uav_speed_m_s': v_uav,
        'smoke_id': 1,
        'drop_x': float(U0[0] + v_uav * Udir[0] * t_drop),
        'drop_y': float(U0[1] + v_uav * Udir[1] * t_drop),
        'drop_z': float(U0[2]),
        'det_x': float(R_expl[0]),
        'det_y': float(R_expl[1]),
        'det_z': float(R_expl[2]),
        'covered_time_s': float(total)
    }]
    df = pd.DataFrame(rows)
    out = "result_q1.xlsx"
    df.to_excel(out, index=False)
    print("Saved", out)

if __name__ == "__main__":
    main()
