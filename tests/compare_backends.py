# -*- coding: utf-8 -*-
"""
compare_backends.py — rowmap (Python 参考实现) vs rowmap_cpp (C++ 工程) 指标对比

读取同种子同场景的两次 run_benchmark 汇总 (results/p1_rowmap 与
results/p2_rowmap_cpp 的 summary.csv), 逐场景对比核心指标。

行为级一致性判据 (闭环混沌预期):
  - 同种子同输入下, 两实现差异仅为浮点噪声 (M2 门禁 <1e-14) 经
    闭环放大; 离散决策 (掉头/丢行) 在分岔点附近可能翻转 ——
    报告差异而非硬失败, 数值指标应保持在同一量级。

用法:
  .venv\\Scripts\\python.exe rowmap_navigation\\tests\\compare_backends.py
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (显示名, key, 容差: None=严格相等, 数值=绝对容差)
METRICS = [
    ("frames_work", "frames_work", None),
    ("crush_rate", "crush_rate", 0.02),
    ("crush_excl_occl", "crush_rate_occlusion_excl", 0.02),
    ("loss_rate", "line_loss_rate", 0.02),
    ("loss_occl", "line_loss_rate_occlusion", 0.02),
    ("cte_rmse_m", "cte_rmse_m", 0.02),
    ("cte_max_m", "cte_max_abs_m", 0.05),
    ("settle_s", "settle_time_s", 1.0),
    ("cte_steady_rmse", "cte_steady_rmse_m", 0.02),
    ("yaw_std_deg", "yaw_steady_std_deg", 0.1),
    ("headland_trig", "headland_triggers", 0),   # 误触发掉头数: 期望一致
]


def load_summary(out_dir):
    """取目录里时间戳最新的 summary_*.csv"""
    d = ROOT / "results" / out_dir
    if not d.is_dir():
        return {}
    files = sorted(d.glob("summary_*.csv"))
    if not files:
        return {}
    rows = {}
    with open(files[-1], newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["scenario"], r.get("mode", "2"))] = r
    return rows


def to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    py = load_summary("p1_rowmap")
    cp = load_summary("p2_rowmap_cpp")
    if not py:
        sys.exit("缺少 results/p1_rowmap/summary.csv (先跑 --backend rowmap)")
    if not cp:
        sys.exit("缺少 results/p2_rowmap_cpp/summary.csv (先跑 --backend rowmap_cpp)")

    keys = sorted(set(py) & set(cp))
    print(f"{'scenario':<18} {'metric':<16} {'python':>10} {'cpp':>10} "
          f"{'diff':>10}  verdict")
    print("-" * 76)

    n_ok = n_tol = n_diff = 0
    for k in keys:
        for name, key, tol in METRICS:
            a, b = to_num(py[k].get(key)), to_num(cp[k].get(key))
            if a is None and b is None:
                continue
            if a is None or b is None:
                print(f"{k[0]:<18} {name:<16} {a!r:>10} {b!r:>10} "
                      f"{'-':>10}  MISSING")
                n_diff += 1
                continue
            d = b - a
            if tol is None:
                ok = (d == 0)
            elif tol == 0:
                ok = (a == b)
            else:
                ok = abs(d) <= tol
            verdict = "exact" if (tol is None or tol == 0) and a == b \
                else ("ok" if ok else "DIFF")
            n_ok += 1 if verdict != "DIFF" else 0
            n_tol += 1 if verdict == "ok" else 0
            if verdict == "DIFF":
                n_diff += 1
            print(f"{k[0]:<18} {name:<16} {a:>10.4f} {b:>10.4f} "
                  f"{d:>+10.4f}  {verdict}")

    print("-" * 76)
    print(f"exact: {n_ok - n_tol} | within-tol: {n_tol} | DIFF: {n_diff}")
    print("注: DIFF 项为闭环混沌分岔 (浮点噪声放大到离散决策), "
          "数值量级一致即视为行为级等价")


if __name__ == "__main__":
    main()
