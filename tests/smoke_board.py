# -*- coding: utf-8 -*-
"""
smoke_board.py — RK3588 板端替换 .so 后的冒烟测试 (M4)

在板端工程根目录运行 (navigation/ 命名空间包可导入处):

    cd /path/to/board_project        # 板端工程根目录 (含 navigation/ 包)
    python3 /path/to/rowmap_navigation/tests/smoke_board.py .

验证链路 (全部走生产同款路径):
  1. import navigation.vision_navigation → 确认加载的是新 .so
     (模块 doc 标识 rowmap_navigation);
  2. utils.params.get_camera_info 读真实 inis/*.ini 构造相机参数
     (与 contexts.py 生产调用完全同款);
  3. 按既有板端接口签名构造 VisionNavigation + set_sensor_mode(2);
  4. 喂 CAN RTK 包 (实车接收机输出字段格式, 50Hz→每帧 3 包);
  5. 合成 mask 模拟车正前方作物行 → 断言建行锁定/quality=95/
     输出量级合理;
  6. 符号约定专项: mask 左移 100px (行在车左侧) → front_dist 应为
     正 (车在线右侧, 右偏为正) —— 装车前最要命的符号检查;
  7. 诊断快照 (fusion_state / rtk_pos_diag / map_stats) 供人工核对。

退出码: 0=全部通过, 1=失败 (细节打印到 stdout)。
仅依赖 numpy 与板端工程自身模块, 无仿真器依赖。
"""

import math
import sys
from pathlib import Path

import numpy as np


def die(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    # ── 0. 板端工程根目录 (navigation/ 所在地) ──
    board_root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if not (board_root / "navigation").is_dir():
        die(f"{board_root} 下无 navigation/ 目录 —— 请在板端工程根目录运行")
    sys.path.insert(0, str(board_root))

    import navigation.vision_navigation as vn_mod  # noqa: E402

    # ── 1. 确认加载的是 rowmap_navigation 新库 ──
    doc = (vn_mod.__doc__ or "").strip()
    print(f"[1] module: {vn_mod.__file__}")
    print(f"    doc: {doc[:60]}")
    if "rowmap" not in doc.lower():
        die("加载的不是 rowmap_navigation 编译产物 (doc 无 rowmap 标识) "
            "—— 检查 navigation/ 里的 .so 是否已替换")

    from utils.params import get_camera_info  # noqa: E402

    # ── 2. 生产同款相机参数 (真实 inis) ──
    cfg = {
        "camera_params_file": "inis/camera_params.ini",
        "navi_params_file": "inis/navi_params.ini",
        "frame_width": 1080, "frame_height": 1280,
        "max_trans_image_height": 800,
        "width_offset_left": 0.5, "width_offset_right": 0.5,
    }
    camera_info, wheelbase = get_camera_info(
        cfg["camera_params_file"], cfg["navi_params_file"],
        cfg["frame_width"], cfg["frame_height"],
        cfg["max_trans_image_height"],
        cfg["width_offset_left"], cfg["width_offset_right"])
    print(f"[2] camera: crop={camera_info.crop_w}x{camera_info.crop_h} "
          f"camera_rad={math.degrees(camera_info.camera_rad):.2f}deg "
          f"wheelbase={wheelbase}m")

    # ── 3. 按既有板端接口签名构造 ──
    vn = vn_mod.VisionNavigation(
        wheelbase_length=wheelbase,
        camera_info=vars(camera_info),
        image_height=camera_info.crop_h,
        vehicle_center_point_x_thresh=0.3,
        test_mode=True)
    vn.set_sensor_mode(2)
    print("[3] VisionNavigation constructed (rowmap C++)")

    # ── 4. RTK 基准 (首包锚定 ENU 原点) ──
    LAT0, LON0 = 30.0, 120.0
    M_DEG = 111320.0
    cos_lat0 = math.cos(math.radians(LAT0))

    def rtk_pkt(t: float, north_m: float, east_m: float) -> dict:
        # 字段与实车接收机输出一致 (heading/roll/pitch 度,
        # None=不可用; 此处全部给有效值)
        return {
            "pos_status": 4, "diff_age_s": 0.4,
            "latitude": LAT0 + north_m / M_DEG,
            "longitude": LON0 + east_m / (M_DEG * cos_lat0),
            "sat_count": 24, "altitude": 10.0,
            "speed_ms": 1.0, "speed_dir": 0.0,
            "roll": 1.0, "pitch": 2.0, "heading": 0.0,
            "recv_time": t,
        }

    # ── 5. 合成 mask: 车正前方竖直行条带 ──
    h, w = camera_info.crop_h, camera_info.crop_w
    cx = float(np.asarray(camera_info.K_final)[0, 2])   # 光轴列 = 车正前

    def synth_mask(center_col: float) -> np.ndarray:
        m = np.zeros((h, w), dtype=np.uint8)
        c = int(round(center_col))
        lo, hi = max(0, c - 20), min(w, c + 20)
        # 只画下半部 (对应车前方近地); 条带宽度近处 ~40px
        m[int(h * 0.55):, lo:hi] = 255
        return m

    class FakeMask:
        """鸭子类型 mask 载体 (生产为 CppYOLOv8SegAdapter 的
        MaskDataPlus, 同样暴露 .mask/.classic_class/.score)"""
        def __init__(self, mask):
            self.mask = mask
            self.classic_class = 0
            self.score = 0.95

    def run_episode(tag: str, col_offset_px: float, frames: int = 30):
        """一段冒烟剧情: 车缓速正北前进, 行条带固定在 (cx+col_offset_px)"""
        vn.force_row_reset()
        north = 0.0
        last = None
        for f in range(frames):
            t = f / 15.0
            north += 1.0 / 15.0            # 1 m/s
            for _ in range(3):              # 50Hz RTK → 每帧 3 包
                vn.feed_rtk_data(rtk_pkt(t, north, 0.0))
            masks = [FakeMask(synth_mask(cx + col_offset_px))]
            black = np.zeros((h, w, 3), dtype=np.uint8)
            last = vn.get_navigation_data(black, masks, t)
        yaw, front, rear, nav_line, q = last
        print(f"[{tag}] quality={q} yaw={yaw:+.3f}deg "
              f"front={front:+.3f}m rear={rear:+.3f}m "
              f"nav_line={len(nav_line)}pts")
        return last

    # 剧情 A: 行在正前方 → 建行/锁定/q=95/输出近零
    yaw_a, front_a, rear_a, line_a, q_a = run_episode("A 正前方", 0.0)
    if q_a != 95:
        die(f"剧情 A quality={q_a} (期望 95) —— 建行或锁定失败, "
            f"见上方诊断")
    for name, v in (("yaw", yaw_a), ("front", front_a), ("rear", rear_a)):
        if not math.isfinite(v):
            die(f"剧情 A {name} 非有限值: {v}")
    if abs(front_a) > 0.5 or abs(yaw_a) > 5.0:
        die(f"剧情 A 量级异常: front={front_a:.3f}m yaw={yaw_a:.3f}deg "
            f"(行在正前方, 期望近零)")

    # 剧情 B: 行左移 100px (行在车左 → 车在线右) → front 应为正 (右偏正)。
    # 竖直条带在地面对应斜线行 (近端偏移小/远端大), 拟合后外推到前轴
    # 量值 ~0.9m 量级 —— 本剧情只验证符号方向, 不验证量值
    yaw_b, front_b, rear_b, line_b, q_b = run_episode("B 行偏左100px", -100.0)
    if q_b != 95:
        die(f"剧情 B quality={q_b} (期望 95)")
    if not (front_b > 0.05):
        die(f"符号约定错误: 行在车左侧时 front={front_b:+.3f}m, "
            f"期望正值 (车在线右侧, 右偏为正) —— 停止装车, 检查符号链")

    # ── 6. 诊断快照 ──
    fusion = vn.get_fusion_state()
    rtkd = vn.get_rtk_position_diag()
    attd = vn.get_attitude_diag()
    stats = vn.map_stats()
    print(f"[6] fusion: mode={fusion['mode']} rtk_enu_active="
          f"{fusion['rtk_enu_active']} map_size={fusion['map_size']}")
    print(f"    rtk: pos_status={rtkd['rtk_pos_status']} "
          f"heading={rtkd['heading_deg']:.2f}deg "
          f"speed={rtkd['rtk_speed_ms']:.2f}m/s")
    print(f"    attitude: enabled={attd['enabled']} has_data="
          f"{attd['has_data']} applied_roll={attd['applied_roll_deg']:.1f}deg "
          f"applied_pitch={attd['applied_pitch_deg']:.1f}deg")
    print(f"    map: n_rows={stats['n_rows']} task_row_id="
          f"{stats['task_row_id']} task_anchors={stats['task_anchors']}")

    if not fusion["rtk_enu_active"]:
        die("rtk_enu_active=False —— RTK 位姿未建立")
    if not attd["has_data"]:
        die("姿态数据缺失 (roll/pitch 未被接收)")

    print("=" * 60)
    print("SMOKE PASS — rowmap_navigation .so 板端冒烟全部通过")
    print("  (符号约定/建行锁定/RTK 位姿/姿态补偿均已验证)")
    sys.exit(0)


if __name__ == "__main__":
    main()
