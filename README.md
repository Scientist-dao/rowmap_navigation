# rowmap_navigation — 行结构地图视觉导航核心

基于行结构地图 (RowMap) 的农业机械视觉导航核心算法, 独立 C++ 实现,
经 pybind11 封装为 Python 扩展模块。编译产物模块名为
`vision_navigation`, 与既有板端集成代码的导入路径
(`from navigation.vision_navigation import VisionNavigation`) 保持
二进制兼容 —— 将本工程编译的 `.so`/`.pyd` 拷贝到板端工程
`navigation/` 目录即完成算法替换。

配套仿真平台 [nav_simulator](../nav_simulator) 内含同一算法的
Python 参考实现 (供交叉验证与快速迭代), 两套实现相互独立,
由数值对照门禁持续验证行为一致。

## 目录结构

```
rowmap_navigation/
├── CMakeLists.txt              # 构建 (PC / RK3588 板端)
├── include/rowmap/
│   ├── row_map.hpp             # 行结构地图 (锚点融合/任务行锁定/GC)
│   ├── projector.hpp           # 地面投影器 (姿态补偿/双向投影, 零 OpenCV)
│   ├── nav_core.hpp            # 导航解算 (RTK 锚定/拟合/质量分级/掉头中止检测)
│   └── mlp.hpp                 # 三层 MLP 原生前向 (.wts 部署权重)
├── src/
│   └── bindings.cpp            # pybind11 绑定 (模块名 vision_navigation)
├── tools/
│   └── export_weights.py       # npz → .wts 部署权重导出器
├── tests/
│   ├── test_row_map_parity.py  # 地图核心数值对照门禁 (逐帧逐项)
│   ├── test_nav_core_parity.py # 全流水数值对照门禁 (场景A/B/C)
│   ├── compare_backends.py     # rowmap vs rowmap_cpp 闭环指标对比
│   └── smoke_board.py          # RK3588 板端替换 .so 后的冒烟测试
├── lib/                        # 编译产物同步目录 (部署源)
└── build-pc/                   # PC 构建树 (生成)
```

## 构建与测试

PC 构建 (Windows, VS 2026 Build Tools):

```powershell
uv sync
$py  = (Resolve-Path ".\.venv\Scripts\python.exe").Path
$pb  = (Resolve-Path ".\.venv\Lib\site-packages\pybind11\share\cmake\pybind11").Path
cmake -S . -B build-pc -G "Visual Studio 18 2026" -A x64 `
    -DPython_EXECUTABLE="$py" -Dpybind11_DIR="$pb"
cmake --build build-pc --config Release
```

RK3588 板端构建 (aarch64, 板上原生编译):

```bash
# 板上依赖: cmake g++ python3-dev + pip3 install pybind11
# (板上主跑 python3.10 —— 与既有部署包的 cpython-310 .so 对应;
#  若还需要其他版本, 换对应 python 重复构建)
cmake -S . -B build-rk3588 \
    -DPython_EXECUTABLE=$(which python3) \
    -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
cmake --build build-rk3588 -j4
# 产物: build-rk3588/vision_navigation.cpython-310-aarch64-linux-gnu.so
# 部署: 备份并替换目标工程 navigation/ 内同名文件, 重启服务
```

板端冒烟 (替换 .so 后, 在目标工程根目录):

```bash
python3 /path/to/rowmap_navigation/tests/smoke_board.py .
# 验证: 新库加载标识 / 真实 inis 构造 / 建行锁定 q=95 /
#       符号约定专项 (行偏左→front 为正) / RTK 位姿 / 姿态补偿
```

数值对照门禁 (每次修改后必跑; 需将 [nav_simulator](../nav_simulator)
仓库与本仓库克隆到同一父目录 —— 门禁从共同父目录导入 Python 参考实现):

```powershell
# 地图核心 —— 160 帧随机观测序列, 锚点 (y,x,var,n) 逐位全等
.\.venv\Scripts\python.exe -X utf8 tests\test_row_map_parity.py

# 导航核心全流水 —— 场景A 280 帧 RTK+mask 序列, 四档 quality 全覆盖
#     (建图/跟踪/遮挡外推/RTK 降级/横漂/行尾越界/掉头引导/复位),
#     离散量严格相等, 连续量容差 1e-9 (BLAS FMA 良性 ULP 差,
#     实测最大差 yaw 2.5e-14 / 锚点相对 1.1e-15);
#     场景B 63 帧掉头中止专项 (TURN 段行延伸 → row_alive 翻 True
#     → 回 WORK 清零, 逐帧比对);
#     场景C 280 帧感知网络部署态对照: 同一 .wts 权重喂双侧
#     (numpy runtime vs mlp.hpp 原生前向), 连续量容差
#     1e-5 (float32 前向累加序噪声, 实测最大 yaw 7.4e-6),
#     离散量严格相等; 权重取自 nav_simulator 仓库 models/
.\.venv\Scripts\python.exe -X utf8 tests\test_nav_core_parity.py
```

仿真器闭环 (在 nav_simulator 仓库根目录):

```powershell
# 后端切换: --backend baseline / rowmap / rowmap_cpp
.\.venv\Scripts\python.exe -m nav_simulator.main -s straight_3row `
    --backend rowmap_cpp --headless --max-frames 450

# 同种子双后端基准 + 指标对比 (6 场景 42 项指标一致)
.\.venv\Scripts\python.exe -m nav_simulator.run_benchmark --backend rowmap_cpp --seeds 1
.\.venv\Scripts\python.exe -m nav_simulator.run_benchmark --backend rowmap --seeds 1
.\.venv\Scripts\python.exe rowmap_navigation\tests\compare_backends.py
```

## 功能阶段

| 阶段 | 内容 | 状态 |
|---|---|---|
| 行结构地图 | row_map.hpp: 锚点融合 / 任务行锁定 / 段查询 / GC + 数值对照门禁 | 完成 |
| 导航解算核心 | projector.hpp + nav_core.hpp + bindings.cpp: 姿态补偿投影 / RTK ENU 锚定 / 前向段拟合 / 质量分级 / 重捕获引导 / 掉头中止检测 (row_alive), 按既有板端接口签名构造 + 全诊断接口绑定 | 完成 |
| 仿真闭环接入 | nav_simulator rowmap_cpp 后端, 6 场景同种子闭环 A/B: 42 项指标 (crush/loss/CTE/掉头触发) 全部一致; hard_gap25 场景 4 种子 4800 帧影子对照无结构差异 (闭环轨迹分岔经分析为浮点舍入混沌放大, 非算法差异) | 完成 |
| 感知网络部署前向 | mlp.hpp 三层 MLP 原生前向 + .wts 部署权重格式 + tools/export_weights.py 导出器; 部署态对照 280 帧全过 (容差 1e-5); 闭环注入验证 (hard_gap25 1200 帧平均质量 75.4 → 85.3, 学习到的方差信息生效) | 完成 |
| RK3588 边缘部署 | aarch64 构建 + 目标工程替换部署 + 冒烟测试 | 冒烟脚本就绪, 待板上执行 |
| 田间实地运行 | — | 待开始 |

## 已知设计边界

- 位姿基准依赖 RTK ENU hint (世界系锚定由首包经纬度建立);
  无 RTK 场景下解算瘫痪 (仿真器侧已自动兜底启用仿真 RTK)。
- 前向段为直线拟合, 强弯道跟踪误差偏大 (curved_sharp 场景
  CTE RMSE ~0.86m); 曲线行支持列入后续版本。
- 地图锚点随里程增长无裁剪上限, 长时间作业行数/锚点累积
  会拖慢逐帧解算 (需时再加窗口裁剪)。
- 质量评估/行尾感知网络输入须落在训练分布内: 训练数据全部
  speed=1.0 m/s + RTK 固定解 (pos_status=4), 特征中相应分量
  feat_std 退化 (~1e-6), 分布外取值会被标准化放大 ~1e6 倍把
  网络推向饱和恒输出; 部署时若作业车速偏离训练值需重训或扩
  特征标准化范围。
