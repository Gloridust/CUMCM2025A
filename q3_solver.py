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
    # prepare rows matching result1 template
    th_rad = math.radians(th_best); ux,uy = math.cos(th_rad), math.sin(th_rad)
    heading_deg = int(round((math.degrees(math.atan2(uy, ux))) % 360))
    rows=[]
    for k, E in enumerate(E_best, start=1):
        Dk = DROP_TIMES[k-1]
        x_e = U0[0] + v_best*ux*E
        y_e = U0[1] + v_best*uy*E
        z_e = 1800.0 - 0.5*g*(E - Dk)**2
        r_drop = U0 + v_best*np.array([ux,uy,0.0]) * Dk
        ti, _ = intervals_for_explosion(E, x_e, y_e, z_e, dt_scan=cfg['DT_FINE'])
        rows.append({
            '无人机运动方向': heading_deg,
            '无人机运动速度 (m/s)': round(v_best,3),
            '烟幕干扰弹编号（1 2 3）': k,
            '烟幕干扰弹投放点的x坐标 (m)': round(r_drop[0],3),
            '烟幕干扰弹投放点的y坐标 (m)': round(r_drop[1],3),
            '烟幕干扰弹投放点的z坐标 (m)': round(r_drop[2],3),
            '烟幕干扰弹起爆点的x坐标 (m)': round(x_e,3),
            '烟幕干扰弹起爆点的y坐标 (m)': round(y_e,3),
            '烟幕干扰弹起爆点的z坐标 (m)': round(z_e,3),
            '有效干扰时长 (s)': round(ti,3)
        })
    # DataFrame 按题目 result1 模板字段保存
    df = pd.DataFrame(rows, columns=[
        '无人机运动方向','无人机运动速度 (m/s)','烟幕干扰弹编号（1 2 3）',
        '烟幕干扰弹投放点的x坐标 (m)','烟幕干扰弹投放点的y坐标 (m)','烟幕干扰弹投放点的z坐标 (m)',
        '烟幕干扰弹起爆点的x坐标 (m)','烟幕干扰弹起爆点的y坐标 (m)','烟幕干扰弹起爆点的z坐标 (m)',
        '有效干扰时长 (s)'
    ])
    out_path = "result1.xlsx"
    df.to_excel(out_path, index=False)
    print("Final union shielding time:", round(tot,3), "s")
    print("Saved result to", out_path)

if __name__ == '__main__':
    main()
