# -*- coding: utf-8 -*-
"""
test_nav_core_parity.py — RowMapNavigation 全流水数值对照门禁 (M2)

同一 RTK 包序列 + 同一 mask 序列喂给:
  - C++ 实现: rowmap_navigation 编译的 vision_navigation.VisionNavigation
  - Python 参考实现: nav_simulator/rowmap_proto/nav_core.RowMapNavigation

逐帧断言输出一致: quality / nav_line_px / 地图离散量 (行数/锁
定/锚点数) 严格相等; 连续量 (yaw/front/rear/var) 容差 1e-9。

容差来源 (良性, 非算法差异): Python 参考实现投影器的矩阵×向量走
OpenBLAS (FMA 收缩, 乘加一次舍入), C++ 朴素循环乘/加分别舍入
——反投影输出存在 ULP 级差 (实测 ~1e-16 相对), 逐帧传播到
锚点与输出。任何真正的移植错误 (公式/参数/顺序) 都在 1e-9
阈值上显形。全程报告实际最大差作为等价精度证据。

覆盖场景 (场景A 280 帧, 车辆沿微弯行推进 70m):
  建图捕获(q→95) / 正常跟踪 / 姿态摆动补偿(roll/pitch 正弦) /
  遮挡外推(q=70) / RTK 降级窗(pos_status=2 位置 hold) /
  横向漂移(CTE 0.4→2.9m) / 横移回收(CAPTURE) /
  行尾越界(WORK: q=50 引导→q=0) / 掉头段引导复活(TURN: q→50) /
  force_row_reset 状态清空
覆盖场景 (场景B 63 帧, 掉头中止信号专项):
  WORK 建图 → TURN_LEFT 期间行前方持续延伸 → row_alive 翻 True
  (任一行跨度较掉头开始快照增长 ≥2m) → 回 WORK 清零。逐帧比对
  row_alive/quality/yaw/front/rear/task_locked。

用法:
  .venv\\Scripts\\python.exe rowmap_navigation\\tests\\test_nav_core_parity.py
"""

import importlib.util
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# 先加载 C++ 模块再 import 参考实现: 防裸名 vision_navigation 在
# sys.modules 里被基线库版本占用 (以 __file__ 路径区分)
sys.path.insert(0, str(ROOT / "rowmap_navigation" / "build-pc" / "Release"))
sys.path.insert(0, str(ROOT))

import vision_navigation as cpp_mod  # noqa: E402
from nav_simulator.rowmap_proto.nav_core import RowMapNavigation as PyNav  # noqa: E402
from nav_simulator.rendering.mask_generator import MaskGenerator  # noqa: E402
from nav_simulator.core.baseline_lib import import_baseline_fixture  # noqa: E402


def _build_ci():
    """相机参数: 基线库 fixture (与板端同源) 优先, 缺失时回退
    nav_simulator 演示参数 (真实标定链路预计算, 数值一致)"""
    fixture = import_baseline_fixture()
    if fixture is not None:
        return fixture()
    from nav_simulator.core.camera_params import build_camera_info
    return build_camera_info()

METRES_PER_DEG = 111320.0
ORIGIN_LAT, ORIGIN_LON = 30.0, 120.0    # 任意基准 (首包锚定, 相对量无关)
FPS = 15.0
N_FRAMES = 280
TOL = 1e-9


def phase_of(frame):
    """帧号 → (ctrl_hint, occluded, pos_status) 阶段编排"""
    if frame < 10:
        return "CAPTURE", False, 4        # 建图捕获
    if frame < 90:
        return "WORK", False, 4           # 正常跟踪 (姿态摆动全程)
    if frame < 105:
        return "WORK", True, 4            # 遮挡外推 → q=70
    if frame < 115:
        return "WORK", False, 2           # RTK 降级: 位置 hold
    if frame < 180:
        return "WORK", False, 4           # 跟踪恢复
    if frame < 200:
        return "WORK", False, 4           # 横向漂移 +2.5m (CTE 拉大)
    if frame < 220:
        return "CAPTURE", False, 4        # 横移回收
    if frame < 260:
        return "WORK", False, 4           # 行尾越界: q=50 引导 → q=0
    return "TURN_LEFT", False, 4          # 掉头段: 引导复活 q→50


def stats_compat(py_stats, cpp_stats, tol=1e-9):
    """两侧 map_stats 相等性 (离散量严格; var_min 相对容差,
    task_row_id: None↔-1 归一)"""
    pid = py_stats["task_row_id"]
    pid = -1 if pid is None else pid
    a, b = py_stats["task_var_min"], cpp_stats["task_var_min"]
    var_eq = ((a != a and b != b)
              or math.isclose(a, b, rel_tol=tol))
    return (py_stats["n_rows"] == cpp_stats["n_rows"]
            and pid == cpp_stats["task_row_id"]
            and py_stats["task_anchors"] == cpp_stats["task_anchors"]
            and py_stats["total_anchors"] == cpp_stats["total_anchors"]
            and var_eq)


def anchors_close(a_py, a_cp, tol=1e-9):
    """锚点元组 (x, y, sigma, n) 比较: 坐标/σ 相对容差, n 严格"""
    return (a_py[3] == a_cp[3]
            and math.isclose(a_py[0], a_cp[0], rel_tol=tol)
            and math.isclose(a_py[1], a_cp[1], rel_tol=tol)
            and math.isclose(a_py[2], a_cp[2], rel_tol=tol))


def main():
    rng = np.random.default_rng(7)

    ci = _build_ci()
    py_nav = PyNav()
    cpp_nav = cpp_mod.VisionNavigation(
        wheelbase_length=2.7, camera_info=vars(ci), image_height=ci.crop_h)
    gen = MaskGenerator(ci)

    # ── 场景: 世界系微弯行 x = 0.30 + 0.002·y, y ∈ [-5, 55] ──
    def row_x(y):
        return 0.30 + 0.002 * y

    east, north = 0.72, -1.0     # 车辆真值 (RTK 基准系), 初始 CTE≈0.42m
    speed = 3.75                  # m/s (Δy=0.25m/帧)

    max_diff = {"yaw": 0.0, "front": 0.0, "rear": 0.0, "anchor": 0.0}
    quality_hist = {}
    row_alive_hist = {}
    fails = []

    for frame in range(N_FRAMES):
        t = frame / FPS
        ctrl, occluded, pos_status = phase_of(frame)

        # 真值推进: 纵向恒速 + 阶段横向漂移 (heading 保持 0, 纯平移)
        north += 0.25
        if 180 <= frame < 200:
            east += 0.125         # 向右横漂 (CTE 0.42 → 2.92m)
        elif 200 <= frame < 220:
            east -= 0.125         # 横移回收
        roll_t = math.radians(2.0) * math.sin(t * 0.7)
        pitch_t = math.radians(1.5) * math.sin(t * 0.5 + 1.0)

        # RTK 包 (位置/航向/姿态加噪; pos_status 按阶段)
        ne, nn = rng.normal(0.0, 0.02, 2)
        cos_lat = math.cos(math.radians(ORIGIN_LAT))
        pkt = {
            "recv_time": t,
            "latitude": ORIGIN_LAT + (north + nn) / METRES_PER_DEG,
            "longitude": ORIGIN_LON + (east + ne) / (METRES_PER_DEG * cos_lat),
            "altitude": 10.0,
            "speed_ms": speed,
            "speed_dir": 0.0,
            "pos_status": pos_status,
            "sat_count": 24,
            "diff_age_s": 0.4,
            "heading": rng.normal(0.0, 0.3),
            "roll": math.degrees(roll_t) + rng.normal(0.0, 0.1),
            "pitch": math.degrees(pitch_t) + rng.normal(0.0, 0.1),
        }
        py_nav.feed_rtk_data(pkt)
        cpp_nav.feed_rtk_data(pkt)
        py_nav.set_speed_hint(speed)
        cpp_nav.set_speed_hint(speed)
        py_nav.set_control_hint(ctrl)
        cpp_nav.set_control_hint(ctrl)

        # mask: 真值姿态成像 (相机随车身倾斜), 行点世界系→车辆系
        if occluded:
            masks = []
        else:
            gen.set_attitude(roll_t, pitch_t)
            pts_vehicle = []
            for wy in np.arange(-5.0, 55.0, 0.5):
                wx = row_x(wy)
                dx, dy = wx - east, wy - north
                xv = dx * 1.0 - dy * 0.0    # heading=0: c=1, s=0
                zv = dx * 0.0 + dy * 1.0
                if 0.5 <= zv <= 24.0:
                    pts_vehicle.append((xv, zv))
            masks = gen.generate_masks(
                [(0, pts_vehicle, 0, 0.30)], use_physical_width=True)

        # 解算 (两侧同输入)
        py_out = py_nav.get_navigation_data(None, masks, t)
        cp_out = cpp_nav.get_navigation_data(None, masks, t)

        # ── 断言 ──
        py_yaw, py_front, py_rear, py_line, py_q = py_out
        cp_yaw, cp_front, cp_rear, cp_line, cp_q = cp_out
        quality_hist[py_q] = quality_hist.get(py_q, 0) + 1

        if py_q != cp_q:
            fails.append(f"frame {frame}: quality {py_q} != {cp_q}")
        if list(map(tuple, py_line)) != list(map(tuple, cp_line)):
            fails.append(f"frame {frame}: nav_line_px {py_line} != {cp_line}")
        for name, a, b in (("yaw", py_yaw, cp_yaw),
                           ("front", py_front, cp_front),
                           ("rear", py_rear, cp_rear)):
            d = abs(a - b)
            if d > max_diff[name]:
                max_diff[name] = d
            if d > TOL:
                fails.append(f"frame {frame}: {name} {a!r} != {b!r} (d={d:.3e})")

        ps, cs = py_nav.map_stats(), cpp_nav.map_stats()
        if not stats_compat(ps, cs):
            fails.append(f"frame {frame}: map_stats {ps} != {cs}")
        if py_nav.task_locked != cpp_nav.task_locked:
            fails.append(f"frame {frame}: task_locked mismatch")
        # 掉头中止信号逐帧对照 (CAPTURE/TURN 段行跨度增长会被触发)
        ra_py = py_nav.get_row_alive()
        ra_cp = cpp_nav.get_row_alive()
        row_alive_hist[ra_py] = row_alive_hist.get(ra_py, 0) + 1
        if ra_py != ra_cp:
            fails.append(f"frame {frame}: row_alive {ra_py} != {ra_cp}")

        # 锚点级对照 (相对容差; 离散量 n/行数严格)
        mo_py = py_nav.get_map_overlay()["rows"]
        mo_cp = cpp_nav.get_map_overlay()["rows"]
        if len(mo_py) != len(mo_cp):
            fails.append(f"frame {frame}: overlay n_rows "
                         f"{len(mo_py)} != {len(mo_cp)}")
        else:
            for rp, rc in zip(mo_py, mo_cp):
                if rp["id"] != rc["id"] or rp["task"] != rc["task"]:
                    fails.append(f"frame {frame}: overlay row meta "
                                 f"{rp['id']}/{rp['task']} != "
                                 f"{rc['id']}/{rc['task']}")
                    continue
                if len(rp["anchors"]) != len(rc["anchors"]):
                    fails.append(f"frame {frame}: row {rp['id']} "
                                 f"anchors {len(rp['anchors'])} != "
                                 f"{len(rc['anchors'])}")
                    continue
                for ap, ac in zip(rp["anchors"], rc["anchors"]):
                    for i in range(3):
                        d = abs(ap[i] - ac[i]) / max(1e-30, abs(ap[i]))
                        if d > max_diff["anchor"]:
                            max_diff["anchor"] = d
                    if not anchors_close(ap, ac):
                        fails.append(f"frame {frame}: row {rp['id']} "
                                     f"anchor {ap} != {ac}")

        if fails:
            print(f"FAIL @ frame {frame} (phase {ctrl}, occluded={occluded})")
            for f in fails[:10]:
                print("  " + f)
            sys.exit(1)

    # ── 尾声: force_row_reset 状态清空 ──
    py_nav.force_row_reset()
    cpp_nav.force_row_reset()
    ps, cs = py_nav.map_stats(), cpp_nav.map_stats()
    reset_ok = (ps["n_rows"] == 0 and cs["n_rows"] == 0
                and ps["total_anchors"] == 0 and cs["total_anchors"] == 0
                and not py_nav.task_locked and not cpp_nav.task_locked
                and not py_nav.get_row_alive()
                and not cpp_nav.get_row_alive())

    # ════════════════════════════════════════════════════════════
    # 场景 B: 掉头中止信号专项对照
    # 语义复现 hard_gap25 误掉头: WORK 沿行建图 25 帧 → TURN_LEFT
    # 36 帧期间行前方持续出新锚点 (行仍在延伸 = 掉头理由消失,
    # 任一行跨度较掉头开始快照增长 ≥2m) → row_alive 翻 True → 回
    # WORK 2 帧清零。主场景 A 建图期跨度增长 ~0.8m 够不着阈值、
    # 行尾后 TURN 段无观测, 故单列此场景非平凡覆盖 True 路径。
    # ════════════════════════════════════════════════════════════
    py_nav = PyNav()
    cpp_nav = cpp_mod.VisionNavigation(
        wheelbase_length=2.7, camera_info=vars(ci), image_height=ci.crop_h)
    east, north = 0.72, -1.0
    nb = 63                    # 0-24 WORK / 25-60 TURN_LEFT / 61-62 WORK
    alive_first = None
    alive_frames_b = 0
    for frame in range(nb):
        t = frame / FPS
        north += 0.25
        ctrl = "WORK" if (frame < 25 or frame >= 61) else "TURN_LEFT"
        roll_t = math.radians(2.0) * math.sin(t * 0.7)
        pitch_t = math.radians(1.5) * math.sin(t * 0.5 + 1.0)
        ne, nn = rng.normal(0.0, 0.02, 2)
        cos_lat = math.cos(math.radians(ORIGIN_LAT))
        pkt = {
            "recv_time": t,
            "latitude": ORIGIN_LAT + (north + nn) / METRES_PER_DEG,
            "longitude": ORIGIN_LON + (east + ne)
                          / (METRES_PER_DEG * cos_lat),
            "altitude": 10.0,
            "speed_ms": speed,
            "speed_dir": 0.0,
            "pos_status": 4,
            "sat_count": 24,
            "diff_age_s": 0.4,
            "heading": rng.normal(0.0, 0.3),
            "roll": math.degrees(roll_t) + rng.normal(0.0, 0.1),
            "pitch": math.degrees(pitch_t) + rng.normal(0.0, 0.1),
        }
        py_nav.feed_rtk_data(pkt)
        cpp_nav.feed_rtk_data(pkt)
        py_nav.set_speed_hint(speed)
        cpp_nav.set_speed_hint(speed)
        py_nav.set_control_hint(ctrl)
        cpp_nav.set_control_hint(ctrl)

        gen.set_attitude(roll_t, pitch_t)
        pts_vehicle = []
        for wy in np.arange(-5.0, 55.0, 0.5):
            wx = row_x(wy)
            dx, dy = wx - east, wy - north
            xv = dx * 1.0 - dy * 0.0
            zv = dx * 0.0 + dy * 1.0
            if 0.5 <= zv <= 24.0:
                pts_vehicle.append((xv, zv))
        masks = gen.generate_masks(
            [(0, pts_vehicle, 0, 0.30)], use_physical_width=True)

        py_yaw, py_front, py_rear, _py_line, py_q = \
            py_nav.get_navigation_data(None, masks, t)
        cp_yaw, cp_front, cp_rear, _cp_line, cp_q = \
            cpp_nav.get_navigation_data(None, masks, t)

        ra_py = py_nav.get_row_alive()
        ra_cp = cpp_nav.get_row_alive()
        if ra_py != ra_cp:
            print(f"FAIL @ 场景B frame {frame} ({ctrl}): "
                  f"row_alive {ra_py} != {ra_cp}")
            sys.exit(1)
        if ra_py:
            if alive_first is None:
                alive_first = frame
            alive_frames_b += 1
        if py_q != cp_q:
            print(f"FAIL @ 场景B frame {frame}: "
                  f"quality {py_q} != {cp_q}")
            sys.exit(1)
        if py_nav.task_locked != cpp_nav.task_locked:
            print(f"FAIL @ 场景B frame {frame}: task_locked mismatch")
            sys.exit(1)
        for name, a, b in (("yaw", py_yaw, cp_yaw),
                           ("front", py_front, cp_front),
                           ("rear", py_rear, cp_rear)):
            if abs(a - b) > TOL:
                print(f"FAIL @ 场景B frame {frame}: "
                      f"{name} {a!r} != {b!r}")
                sys.exit(1)

    assert alive_frames_b > 0, \
        "场景B未触发 row_alive=True (中止信号 True 路径覆盖缺失)"
    assert (not py_nav.get_row_alive()
            and not cpp_nav.get_row_alive()), \
        "场景B结尾 WORK 帧未清零 row_alive"

    # ════════════════════════════════════════════════════════════
    # 场景 C: C3 质量网络 + C5 行尾网络部署态对照
    # 同一 RTK+mask 流 + 同一训练权重喂双侧 (py numpy runtime vs
    # C++ 原生前向): C3 逐 band (δx,σ²) 替代手写三项式 var 进观测
    # 融合, C5 每帧 P(end)。活性证明: 无网络参考实例同喂前 40 帧,
    # 断言输出与网络路径可区分 (证明 C3 确实改变了 var/δx)。
    #
    # 容差 (探针 _probe_mlp_noise.py 标定): float32 前向 BLAS/朴素
    # 累加序差在 logit 上达 ~4e-6 绝对 (实测 max 3.8e-6), exp 传播后
    # var 相对差同量级; 融合+EMA 阻尼后导航输出实测 max ~1.3e-6
    # (行尾引导段锚点稀疏, 阻尼最弱)。统一容差 1e-5, 对两类噪声
    # 均有 ≥2.5x 裕量; 真正的移植错误 (特征错位/算式差) 是 O(1)
    # 量级, 不会被该容差掩盖。离散量 (quality/锁定/锚点数/整型
    # 像素) 仍严格。
    # ════════════════════════════════════════════════════════════
    # 权重目录兼容两种布局: 开发工作区 (本测试所在仓库与
    # nav_simulator 同级, 权重在共同父目录 models/) 或同级克隆
    # (权重在 nav_simulator 仓库根 models/)
    models_dir = next((d for d in (ROOT / "models",
                                   ROOT / "nav_simulator" / "models")
                       if (d / "c3_quality_weights.npz").is_file()), None)
    c3_npz = str(models_dir / "c3_quality_weights.npz") if models_dir else ""
    c5_npz = str(models_dir / "c5_headland_weights.npz") if models_dir else ""
    net_summary = None
    if os.path.isfile(c3_npz) and os.path.isfile(c5_npz):
        # 导出器按文件路径加载 (不经 sys.path, 防与基线库包名冲突)
        spec = importlib.util.spec_from_file_location(
            "rm_export_weights",
            ROOT / "rowmap_navigation" / "tools" / "export_weights.py")
        exp_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp_mod)
        tmpdir = tempfile.mkdtemp(prefix="rmw_parity_")
        c3_wts = os.path.join(tmpdir, "c3.wts")
        c5_wts = os.path.join(tmpdir, "c5.wts")
        exp_mod.npz_to_wts(c3_npz, c3_wts)
        exp_mod.npz_to_wts(c5_npz, c5_wts)

        from nav_simulator.dl.c3_runtime import load_quality_net
        from nav_simulator.dl.c5_runtime import load_headland_net

        py_nav = PyNav()
        py_nav.set_quality_net(load_quality_net(c3_npz))
        py_nav.set_headland_net(load_headland_net(c5_npz))
        cpp_nav = cpp_mod.VisionNavigation(
            wheelbase_length=2.7, camera_info=vars(ci),
            image_height=ci.crop_h)
        assert cpp_nav.set_quality_net(c3_wts), "C++ 加载 C3 权重失败"
        assert cpp_nav.set_headland_net(c5_wts), "C++ 加载 C5 权重失败"

        py_ref = PyNav()          # 无网络参考 (活性证明用)
        ref_diff_max = 0.0
        prob_frames = 0
        probs_c = []              # P(end) 分布诊断 (饱和性核查)
        NET_TOL = 1e-5            # float32 前向累加序噪声 (探针标定)
        nc = N_FRAMES
        max_diff_c = {"yaw": 0.0, "front": 0.0, "rear": 0.0,
                      "anchor": 0.0, "prob": 0.0}
        quality_hist_c = {}
        east, north = 0.72, -1.0
        # 网络喂入须在训练分布内: 训练数据全部 speed=1.0 + pos_status=4
        # (C5 idx18 车速 / C3 idx13 固定解的 feat_std=1e-6 —— 偏离训练
        # 均值的取值被标准化放大 ~1e6 倍, 把网络推向饱和恒输出
        # P(end)≡1; 见 results/_probe_c5_feat.py)。speed hint 同时决定
        # 航向 EMA α, 一并把 C3 idx7/C5 idx19 航向 σ 拉回分布内。
        # 真值运动速度 (3.75 m/s) 与阶段编排不变; 位置 hold 路径由
        # 场景A (无网络) 覆盖。
        speed_hint = 1.0
        for frame in range(nc):
            t = frame / FPS
            ctrl, occluded, _ps = phase_of(frame)
            pos_status = 4
            north += 0.25
            if 180 <= frame < 200:
                east += 0.125
            elif 200 <= frame < 220:
                east -= 0.125
            roll_t = math.radians(2.0) * math.sin(t * 0.7)
            pitch_t = math.radians(1.5) * math.sin(t * 0.5 + 1.0)
            ne, nn = rng.normal(0.0, 0.02, 2)
            cos_lat = math.cos(math.radians(ORIGIN_LAT))
            pkt = {
                "recv_time": t,
                "latitude": ORIGIN_LAT + (north + nn) / METRES_PER_DEG,
                "longitude": ORIGIN_LON + (east + ne)
                              / (METRES_PER_DEG * cos_lat),
                "altitude": 10.0,
                "speed_ms": speed,
                "speed_dir": 0.0,
                "pos_status": pos_status,
                "sat_count": 24,
                "diff_age_s": 0.4,
                "heading": rng.normal(0.0, 0.3),
                "roll": math.degrees(roll_t) + rng.normal(0.0, 0.1),
                "pitch": math.degrees(pitch_t) + rng.normal(0.0, 0.1),
            }
            py_nav.feed_rtk_data(pkt)
            cpp_nav.feed_rtk_data(pkt)
            py_nav.set_speed_hint(speed_hint)
            cpp_nav.set_speed_hint(speed_hint)
            py_nav.set_control_hint(ctrl)
            cpp_nav.set_control_hint(ctrl)
            if frame < 40:
                py_ref.feed_rtk_data(pkt)
                py_ref.set_speed_hint(speed_hint)
                py_ref.set_control_hint(ctrl)

            if occluded:
                masks = []
            else:
                gen.set_attitude(roll_t, pitch_t)
                pts_vehicle = []
                for wy in np.arange(-5.0, 55.0, 0.5):
                    wx = row_x(wy)
                    dx, dy = wx - east, wy - north
                    xv = dx * 1.0 - dy * 0.0
                    zv = dx * 0.0 + dy * 1.0
                    if 0.5 <= zv <= 24.0:
                        pts_vehicle.append((xv, zv))
                masks = gen.generate_masks(
                    [(0, pts_vehicle, 0, 0.30)], use_physical_width=True)

            py_out = py_nav.get_navigation_data(None, masks, t)
            cp_out = cpp_nav.get_navigation_data(None, masks, t)
            if frame < 40:
                rf = py_ref.get_navigation_data(None, masks, t)
                ref_diff_max = max(ref_diff_max,
                                   abs(rf[1] - py_out[1]))

            py_yaw, py_front, py_rear, py_line, py_q = py_out
            cp_yaw, cp_front, cp_rear, cp_line, cp_q = cp_out
            quality_hist_c[py_q] = quality_hist_c.get(py_q, 0) + 1
            if py_q != cp_q:
                print(f"FAIL @ 场景C frame {frame}: "
                      f"quality {py_q} != {cp_q}")
                sys.exit(1)
            if list(map(tuple, py_line)) != list(map(tuple, cp_line)):
                print(f"FAIL @ 场景C frame {frame}: nav_line "
                      f"{py_line} != {cp_line}")
                sys.exit(1)
            for name, a, b in (("yaw", py_yaw, cp_yaw),
                               ("front", py_front, cp_front),
                               ("rear", py_rear, cp_rear)):
                d = abs(a - b)
                if d > max_diff_c[name]:
                    max_diff_c[name] = d
                if d > NET_TOL:
                    print(f"FAIL @ 场景C frame {frame}: "
                          f"{name} {a!r} != {b!r} (d={d:.3e})")
                    sys.exit(1)

            # C5 行尾概率逐帧对照 (None 语义 + 相对容差)
            pp = py_nav.get_headland_prob()
            cp = cpp_nav.get_headland_prob()
            if (pp is None) != (cp is None):
                print(f"FAIL @ 场景C frame {frame}: "
                      f"headland_prob None 不一致 {pp} vs {cp}")
                sys.exit(1)
            if pp is not None:
                prob_frames += 1
                probs_c.append(pp)
                d = abs(pp - cp) / max(1e-30, abs(pp))
                if d > max_diff_c["prob"]:
                    max_diff_c["prob"] = d
                if d > NET_TOL:
                    print(f"FAIL @ 场景C frame {frame}: "
                          f"headland_prob {pp!r} != {cp!r}")
                    sys.exit(1)

            ps, cs = py_nav.map_stats(), cpp_nav.map_stats()
            if not stats_compat(ps, cs, tol=NET_TOL):
                print(f"FAIL @ 场景C frame {frame}: "
                      f"map_stats {ps} != {cs}")
                sys.exit(1)
            if py_nav.task_locked != cpp_nav.task_locked:
                print(f"FAIL @ 场景C frame {frame}: task_locked")
                sys.exit(1)

            mo_py = py_nav.get_map_overlay()["rows"]
            mo_cp = cpp_nav.get_map_overlay()["rows"]
            if len(mo_py) != len(mo_cp):
                print(f"FAIL @ 场景C frame {frame}: overlay n_rows")
                sys.exit(1)
            for rp, rc in zip(mo_py, mo_cp):
                if rp["id"] != rc["id"] or len(rp["anchors"]) != \
                        len(rc["anchors"]):
                    print(f"FAIL @ 场景C frame {frame}: "
                          f"row {rp['id']} anchors 数量")
                    sys.exit(1)
                for ap, ac in zip(rp["anchors"], rc["anchors"]):
                    if not anchors_close(ap, ac, tol=NET_TOL):
                        print(f"FAIL @ 场景C frame {frame}: "
                              f"row {rp['id']} anchor {ap} != {ac}")
                        sys.exit(1)
                    for i in range(3):
                        d = abs(ap[i] - ac[i]) / max(1e-30, abs(ap[i]))
                        if d > max_diff_c["anchor"]:
                            max_diff_c["anchor"] = d

        assert ref_diff_max > 1e-9, \
            f"C3 网络未生效? 前 40 帧输出与手写 var 路径无差异 " \
            f"(max front diff {ref_diff_max:.2e})"
        assert prob_frames > 0, "C5 概率全程未出现"
        net_summary = (nc, max_diff_c, quality_hist_c, prob_frames,
                       ref_diff_max, probs_c)
    else:
        print("警告: models/ 训练权重缺失, 场景C (网络对照) 跳过")

    # 场景编排自检: 四档 quality 全部出现
    levels = sorted(quality_hist.keys())

    frames_desc = f"{N_FRAMES}+{nb}"
    scen_desc = "场景A+场景B"
    if net_summary is not None:
        frames_desc += f"+{net_summary[0]}"
        scen_desc += "+场景C"

    print("=" * 64)
    print("PASS: C++ VisionNavigation ↔ Python 参考实现全流水对照")
    print(f"  帧数 {frames_desc} ({scen_desc}) | 容差 {TOL:.0e} | "
          f"实际最大差: yaw={max_diff['yaw']:.2e} "
          f"front={max_diff['front']:.2e} "
          f"rear={max_diff['rear']:.2e} "
          f"anchor_rel={max_diff['anchor']:.2e}")
    if net_summary is not None:
        _, md_c, qh_c, pf, rdm, probs_c = net_summary
        print(f"  场景C (C3+C5 部署态, 容差 1e-5): 最大差 "
              f"yaw={md_c['yaw']:.2e} front={md_c['front']:.2e} "
              f"rear={md_c['rear']:.2e} anchor_rel={md_c['anchor']:.2e} "
              f"prob_rel={md_c['prob']:.2e}")
        # P(end) 分布: 训练充分收敛的网络 logit 常达 ±40+, float32
        # sigmoid 饱和到精确 0.0/1.0 → 两侧逐位相等 (prob_rel=0 的
        # 根因); 非饱和帧才有 1e-7 量级差
        n_sat = sum(1 for p in probs_c if p in (0.0, 1.0))
        print(f"  C5 概率帧数 {pf} | P(end) ∈ "
              f"[{min(probs_c):.3g}, {max(probs_c):.3g}] "
              f"| 饱和(0/1)帧 {n_sat}/{len(probs_c)} | "
              f"C3 活性证明 ref_diff={rdm:.2e} | "
              "quality 分布(场景C): "
              + " ".join(f"q={q}:{qh_c[q]}帧" for q in sorted(qh_c)))
    print(f"  quality 分布(场景A): "
          + " ".join(f"q={q}:{quality_hist[q]}帧" for q in levels))
    print(f"  row_alive: 场景A True:{row_alive_hist.get(True, 0)}帧 | "
          f"场景B 首帧 True=f{alive_first}, 共 {alive_frames_b} 帧, "
          f"结尾 WORK 清零 (两侧逐帧相等)")
    print(f"  地图终态: n_rows={ps['n_rows']} "
          f"total_anchors={cs['total_anchors']} (逐帧严格相等)")
    print(f"  force_row_reset 状态清空: "
          f"{'PASS' if reset_ok else 'FAIL'}")
    assert reset_ok, "force_row_reset 未清空状态"
    assert set(levels) >= {0, 50, 70, 95}, \
        f"场景未覆盖全部 quality 档: {levels}"


if __name__ == "__main__":
    main()
