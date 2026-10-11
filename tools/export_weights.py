# -*- coding: utf-8 -*-
"""export_weights.py — 训练落盘 npz → C++ 部署态 .wts 权重导出

C++ 侧 (include/rowmap/mlp.hpp) 不解析 zip/npz (避免 zlib 依赖与
板端交叉编译麻烦), 部署走自定义小端二进制 .wts:
    magic "RMW1" | u32 version=1 | u32 n_tensors
    每张量: u32 name_len | name | u32 ndim | u32 dims[ndim] | f32 data
张量名与 npz 键一致: W1/b1/W2/b2/W3/b3/feat_mean/feat_std。

用法:
  # 默认: 导出 C3 + C5 到 models/*.wts (npz 同名替换扩展名)
  .venv/Scripts/python.exe rowmap_navigation/tools/export_weights.py

  # 指定网络与路径
  .venv/Scripts/python.exe rowmap_navigation/tools/export_weights.py \\
      --c3 models/c3_quality_weights.npz --c5 models/c5_headland_weights.npz

逐项对照门禁 (test_nav_core_parity.py 场景C) 复用 npz_to_wts()。
"""

import argparse
import os
import struct
import sys

import numpy as np

_TENSORS = ("W1", "b1", "W2", "b2", "W3", "b3",
            "feat_mean", "feat_std")
_MAGIC = b"RMW1"
_VERSION = 1


def npz_to_wts(npz_path: str, wts_path: str) -> str:
    """npz (W1/b1/.../feat_std) → .wts; 返回 wts 路径"""
    z = np.load(npz_path)
    missing = [k for k in _TENSORS if k not in z]
    if missing:
        raise KeyError(f"{npz_path} 缺张量: {missing}")
    with open(wts_path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<II", _VERSION, len(_TENSORS)))
        for name in _TENSORS:
            a = np.ascontiguousarray(z[name], dtype="<f4")
            nb = name.encode("ascii")
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", a.ndim))
            f.write(struct.pack(f"<{a.ndim}I", *a.shape))
            f.write(a.tobytes())
    return wts_path


def main() -> int:
    ws = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))          # 工作区根
    ap = argparse.ArgumentParser(description="npz → .wts 权重导出")
    ap.add_argument("--c3", default=os.path.join(
        ws, "models", "c3_quality_weights.npz"))
    ap.add_argument("--c5", default=os.path.join(
        ws, "models", "c5_headland_weights.npz"))
    ap.add_argument("--out-dir", default=None,
                    help="输出目录 (默认与各自 npz 同目录)")
    args = ap.parse_args()

    rc = 0
    for tag, path in (("C3", args.c3), ("C5", args.c5)):
        if not os.path.isfile(path):
            print(f"[{tag}] 权重缺失, 跳过: {path}")
            rc = 1
            continue
        out = path if args.out_dir is None else os.path.join(
            args.out_dir, os.path.basename(path))
        out = os.path.splitext(out)[0] + ".wts"
        npz_to_wts(path, out)
        z = np.load(path)
        print(f"[{tag}] {path} -> {out} "
              f"(in={z['W1'].shape[1]} h1={z['W1'].shape[0]} "
              f"h2={z['W3'].shape[1]} out={z['W3'].shape[0]})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
