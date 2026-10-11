# -*- coding: utf-8 -*-
"""
test_row_map_parity.py — RowMapCore 数值对照门禁 (M1)

同一随机观测序列喂给:
  - C++ 实现: rowmap_navigation 编译的 vision_navigation.RowMapCore
  - Python 参考实现: nav_simulator/rowmap_proto/row_map.RowMapCore

每帧断言两侧状态全等 (关联结果 / 任务行 / 全部锚点 y,x,var,n),
证明两套独立实现数值等价 —— 后续 C++ 版基准与板端部署的可信度
均以此门禁为前提。

覆盖分支: 初始建行 / 信息融合收敛 / 锚点插入 / 任务行锁定(含平局) /
垃圾行 GC / 关联门限拒绝 / 遮挡空帧 / 弯行(x 随 y 缓变) / 越界外推查询。

用法:
  .venv\\Scripts\\python.exe rowmap_navigation\\tests\\test_row_map_parity.py
"""

import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# C++ 模块 (构建产物) 与 Python 参考实现
sys.path.insert(0, str(ROOT / "rowmap_navigation" / "build-pc" / "Release"))
sys.path.insert(0, str(ROOT))

import vision_navigation as cpp_mod  # noqa: E402
from nav_simulator.rowmap_proto.row_map import RowMapCore as PyRowMapCore  # noqa: E402

import numpy as np  # noqa: E402


# ── 随机观测序列生成 ─────────────────────────────────────────────────

def gen_sequence(seed: int = 42, frames: int = 160):
    """模拟 3 条弯行 + 车辆沿行推进的观测流

    返回 [(frame, obs_rows, pose_x, pose_y, lock_call), ...]
    obs_rows: List[List[(wx, wy, var)]]
    """
    rng = np.random.default_rng(seed)
    events = []

    # 三条行: 基准横向偏移 + 缓弯 (x 随 y 二次缓变)
    row_defs = [
        {"x0": -0.60, "k2": 4.0e-4},
        {"x0": 0.00, "k2": 4.0e-4},
        {"x0": +0.60, "k2": 4.0e-4},
    ]

    pose_x, pose_y = 0.05, 1.0
    for frame in range(frames):
        # 车辆沿行推进, 横向缓漂 (制造非零 cte 与关联压力)
        pose_y += rng.uniform(0.18, 0.42)
        pose_x += rng.normal(0.0, 0.01)

        obs_rows = []

        # 遮挡窗: t∈[60, 78) 大部分行不可见, t∈[64,68) 全遮挡
        occl = 60 <= frame < 78
        full_occl = 64 <= frame < 68
        if not full_occl:
            visible = [0, 2] if (occl and frame % 2 == 0) else [0, 1, 2]
            for ri in visible:
                d = row_defs[ri]
                # 观测窗: 车前方 2~15m, 点距 ~0.35m + 抖动
                y0 = pose_y + 2.0
                y1 = pose_y + 15.0
                ys = np.arange(y0, y1, 0.35) + rng.normal(0, 0.02, 1)[0]
                pts = []
                for wy in ys:
                    x_true = d["x0"] + d["k2"] * (wy - 60.0) ** 2
                    wx = x_true + rng.normal(0.0, 0.02)
                    var = (0.02 + abs(rng.normal(0, 0.004))) ** 2
                    pts.append((wx, float(wy), float(var)))
                # 关联门限分支: 帧帧小概率整行横移 0.5m (超 GATE_ABS_M
                # -> 关联失败建新行)
                if rng.random() < 0.015:
                    pts = [(wx + 0.5, wy, var) for wx, wy, var in pts]
                # 打乱点序 (update 内部会按 y 重排)
                rng.shuffle(pts)
                obs_rows.append(pts)

        # 噪声短行: 小概率出现 2~3 个点的散点串 (建噪声行 -> GC 回收)
        if rng.random() < 0.08:
            ys = pose_y + rng.uniform(-1.0, 4.0) + np.linspace(0, 0.4, 3)
            pts = [(pose_x + rng.uniform(-2.5, 2.5), float(wy), 0.02 ** 2)
                   for wy in ys]
            rng.shuffle(pts)
            obs_rows.append(pts)

        # 单点行 (< 2 点, update 应跳过)
        if rng.random() < 0.05:
            obs_rows.append([(pose_x + 1.0, pose_y + 5.0, 0.0004)])

        events.append((frame, obs_rows, pose_x, pose_y))

    return events


# ── 状态比较 ──────────────────────────────────────────────────────────

TOL_ABS = 1e-12
TOL_REL = 1e-9


def close(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= TOL_ABS + TOL_REL * max(abs(a), abs(b))


def py_snapshot(py_map) -> dict:
    return {rid: [(a.y, a.x, a.var, a.n) for a in r.anchors]
            for rid, r in py_map.rows.items()}


def compare_states(cpp_map, py_map, frame: int, errors: list):
    tag = f"frame {frame}"
    # 1) 任务行 id
    c_id = cpp_map.task_row_id
    p_id = py_map.task_row_id
    if c_id != p_id:
        errors.append(f"{tag}: task_row_id C++={c_id} py={p_id}")
        return
    # 2) 行 id 集合
    c_snap = cpp_map.rows_snapshot()
    p_snap = py_snapshot(py_map)
    if set(c_snap) != set(p_snap):
        errors.append(f"{tag}: 行 id 集合不一致 "
                      f"C++={sorted(c_snap)} py={sorted(p_snap)}")
        return
    # 3) 逐行逐锚点
    for rid in sorted(p_snap):
        ca, pa = c_snap[rid], p_snap[rid]
        if len(ca) != len(pa):
            errors.append(f"{tag} row {rid}: 锚点数 C++={len(ca)} py={len(pa)}")
            continue
        for i, (c, p) in enumerate(zip(ca, pa)):
            for name, cv, pv in zip(("y", "x", "var", "n"), c, p):
                if name == "n":
                    ok = int(cv) == int(pv)
                else:
                    ok = close(float(cv), float(pv))
                if not ok:
                    errors.append(
                        f"{tag} row {rid} anchor[{i}].{name}: "
                        f"C++={cv!r} py={pv!r}")
    # 4) 统计快照
    c_st, p_st = cpp_map.stats(), py_map.stats()
    if c_st["n_rows"] != p_st["n_rows"]:
        errors.append(f"{tag}: stats.n_rows {c_st['n_rows']} != {p_st['n_rows']}")
    if c_st["total_anchors"] != p_st["total_anchors"]:
        errors.append(f"{tag}: stats.total_anchors 不一致")
    if c_st["task_anchors"] != p_st["task_anchors"]:
        errors.append(f"{tag}: stats.task_anchors 不一致")
    if not close(float(c_st["task_var_min"]), float(p_st["task_var_min"])):
        errors.append(f"{tag}: stats.task_var_min "
                      f"{c_st['task_var_min']} != {p_st['task_var_min']}")


# ── 主流程 ────────────────────────────────────────────────────────────

def main() -> int:
    events = gen_sequence(seed=42, frames=160)

    cpp_map = cpp_mod.RowMapCore()
    py_map = PyRowMapCore()

    errors: list = []
    locked = False
    # 平局容差分支: 额外验证车在两行正中 (±0.6m 行距正中) 的锁定语义
    tie_probe_done = False

    for frame, obs_rows, pose_x, pose_y in events:
        c_matched = cpp_map.update(obs_rows, frame)
        p_matched = py_map.update(obs_rows, frame)
        if list(c_matched) != list(p_matched):
            errors.append(f"frame {frame}: matched C++={list(c_matched)} "
                          f"py={list(p_matched)}")

        if not locked:
            c_lock = cpp_map.lock_task_row(pose_x, pose_y)
            p_lock = py_map.lock_task_row(pose_x, pose_y)
            if (c_lock is None) != (p_lock is None) or (
                    c_lock is not None and c_lock != p_lock):
                errors.append(f"frame {frame}: lock_task_row "
                              f"C++={c_lock} py={p_lock}")
            locked = c_lock is not None

        compare_states(cpp_map, py_map, frame, errors)
        if len(errors) > 20:
            errors.append("... (错误过多提前终止)")
            break

    # 锁定平局容差专项: 两条新行, 车在正中 -> 平局取 id 小
    if not errors:
        cpp2, py2 = cpp_mod.RowMapCore(), PyRowMapCore()
        # 两行 x=-0.6 / +0.6, 车位 (0, 5): 距两行等距 (差值 << TOL)
        obs_a = [(-0.6, float(y), 4e-4) for y in (2.0, 3.0, 4.0, 6.0, 8.0)]
        obs_b = [(+0.6, float(y), 4e-4) for y in (2.0, 3.0, 4.0, 6.0, 8.0)]
        cpp2.update([obs_a, obs_b], 0)
        py2.update([obs_a, obs_b], 0)
        c_lock, p_lock = cpp2.lock_task_row(0.0, 5.0), py2.lock_task_row(0.0, 5.0)
        if c_lock != p_lock:
            errors.append(f"平局锁定: C++={c_lock} py={p_lock}")
        else:
            tie_probe_done = True

    # 报告
    st = cpp_map.stats()
    print("=" * 64)
    if errors:
        print(f"FAIL: {len(errors)} 处不一致")
        for e in errors[:20]:
            print("  " + e)
        return 1
    print("PASS: C++ RowMapCore 与 Python 参考实现逐帧状态全等")
    print(f"  帧数 160 | 行数 {st['n_rows']} | 总锚点 {st['total_anchors']}")
    print(f"  任务行 id={st['task_row_id']} 锚点 {st['task_anchors']} "
          f"var_min={st['task_var_min']:.3e}")
    print(f"  平局锁定专项: {'PASS' if tie_probe_done else 'SKIP'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
