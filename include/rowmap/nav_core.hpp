// ============================================================================
// rowmap/nav_core.hpp — RowMap 导航核心 (C++ 版, M2)
//
// 每帧 get_navigation_data 算法流水:
//   1. 位姿: RTK 包 (feed_rtk_data 累积) → ENU 位置 + 双天线航向 EMA。
//      首包 lat/lon 为基准原点 —— 地图活在"RTK 基准系"里, 基准偏移
//      不影响相对量 (CTE/yaw), 无需绝对定位。
//   2. 观测提取: mask 按图像行扫描取列均值 → 反投影车辆系 (外参含
//      IMU 姿态补偿, mode>=1) → 变换世界系点串, 方差随距离增长。
//   3. 地图更新: 行级关联 (最近行, 门限 0.35m) + 锚点 1/var 融合;
//      无观测帧 (遮挡/断垄) 跳过 —— 地图保持。
//   4. 解算: 任务行前段锚点 (按航向方向选段, 支持沿行双向作业) →
//      车辆系 → 加权二次曲线拟合 (c≈0 时自动退化为直线, 直行/弯道
//      统一表达) → CTE/yaw。解算输入不含本帧图像,
//      感知中断时由地图+RTK 外推。
//   5. quality: 命中任务行=95; 中断但地图充足=70 (nav_valid 保持,
//      不触发地头误掉头); 前向拟合失败但地图存在=50 (重捕获引导);
//      地图不足=0。
//   6. 感知网络旁路 (可开关, 默认关): C3 质量网逐 band 前向得
//      (δx, σ²) 替代手写三项式 var; C5 行尾网每帧由地图历史特征
//      前向 P(行终止)。学习只生产观测/概率, 决策仍由几何核心做。
//
// 数值语义要点: heading EMA 取模语义、输出端单极点平滑、
// 前向段按 cos(heading) 选向。
// 本类为纯 C++ (无 pybind 依赖)。
//
// 既有板端库兼容接口 (板端替换 .so 所需): set_debug / set_debug_verbose /
// set_sensor_mode / set_lever_arm / set_lookahead_time /
// get_navigation_data / get_overlay_data / get_fusion_state /
// get_rtk_position_diag / get_attitude_diag / get_last_source_mode /
// motion_speed_valid / motion_speed_ms / feed_rtk_data /
// force_row_reset / set_nav_engaged / set_nav_mode。
// ============================================================================

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "mlp.hpp"
#include "projector.hpp"
#include "row_map.hpp"

namespace rowmap {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegPerRad = 180.0 / kPi;
constexpr double kRadPerDeg = kPi / 180.0;
constexpr double kMetresPerDeg = 111320.0;
// C3 网络输出 σ² 的限幅 (与训练侧 hetero_nll 的 var clamp 一致)
constexpr double kQnetVarMin = 1e-8;
constexpr double kQnetVarMax = 1e4;

// 取模 (结果符号随除数, floored modulo), heading wrap 用
inline double pymod(double a, double b) {
    double r = std::fmod(a, b);
    if (r != 0.0 && ((r < 0.0) != (b < 0.0))) r += b;
    return r;
}

// ── 导航参数 (集中常量, 调参入口) ──
struct NavParams {
    double k_pix_sigma       = 0.005;  // 像素/分割横向噪声: 每 m 前向距离的 σ (m)
    double sigma_rtk         = 0.02;   // RTK 位置噪声 (m)
    double sigma_heading_meas = 0.5 * kRadPerDeg;  // 双天线航向单包噪声 (rad)
    double heading_ema_alpha = 0.15;   // 航向 EMA 平滑系数
    double output_ema_alpha  = 0.4;    // 输出端 (yaw/front/rear) 单极点平滑系数
    int    scan_step_px      = 8;      // mask 图像行扫描采样步长 (px)
    double z_min_obs         = 0.3;    // 观测前向距离范围 (m)
    double z_max_obs         = 20.0;
    double lookahead_m       = 10.0;   // 解算取任务行前段长度 (m)
    double z_front           = 1.35;   // 前轴参考距离 (半轴距; 由构造轴距覆盖)
    double z_rear            = -1.35;
    int    quality_tracking     = 95;  // 本帧观测命中任务行
    int    quality_dead_reckon = 70;   // 感知中断但地图充足
    int    quality_recapture    = 50;  // 前向拟合失败但地图存在: 查表引导
    int    recapture_anchors    = 8;   // 引导取最近锚点邻域的个数
    int    min_solve_anchors    = 2;   // 解算最少锚点数
    double seg_back_m           = 0.5; // 前向段向车身后多取的长度 (m)
    int    min_display_anchors  = 3;   // overlay 上图的最少锚点数 (滤噪声碎行)
    double qnet_max_dx          = 0.15;  // C3 网络 δx 纠偏安全限幅 (m)
};

// ── RTK/IMU 数据包 (feed_rtk_data 输入; 不可用字段为 NaN) ──
struct RtkPacket {
    double latitude  = 0.0;    // 纬度 (度) [必填]
    double longitude = 0.0;    // 经度 (度) [必填]
    double altitude  = 0.0;    // 海拔 (m)
    double speed_ms  = 0.0;    // 速度 (m/s, 前进为正)
    double speed_dir = 0.0;    // 速度方向 (度, 真北)
    double recv_time = 0.0;    // Unix 时间戳 (秒)
    int    pos_status = 0;     // 0无效 1单点 2SBAS 4固定 5浮点
    int    sat_count  = 0;
    double diff_age_s = 0.0;   // 差分龄 (秒, 0=未知)
    double roll_deg   = std::nan("");   // 横滚 (度) [NaN=不可用]
    double pitch_deg  = std::nan("");   // 俯仰 (度)
    double heading_deg = std::nan("");  // 双天线航向 (度)
};

// 输入 mask 视图: 连续 uint8 单通道 2D 缓冲 (绑定层保证生命周期)
struct MaskView {
    const std::uint8_t* data = nullptr;
    int rows = 0;
    int cols = 0;
    float score = 1.0f;      // 分割置信度 (绑定层缺省 1.0, C3 特征用)
};

// 每帧导航输出
struct NavOutput {
    double yaw_deg = 0.0;      // 偏左为负 (罗盘系, 车航向 − 线方向)
    double front_dist = 0.0;   // 前轴横向偏差 (右偏为正)
    double rear_dist = 0.0;    // 后轴横向偏差 (右偏为正)
    std::vector<std::pair<int, int>> nav_line_px;  // 导航线像素端点
    int quality = 0;           // 0/50/70/95
};

// GUI 叠层数据 (get_overlay_data)
struct OverlayData {
    std::vector<std::vector<std::pair<int, int>>> center_lines;
    std::vector<bool> line_used;
    bool valid = false;   // 无效帧清空
};

// 俯视图地图可视化 (get_map_overlay)
struct MapOverlayRow {
    int id = 0;
    bool task = false;
    std::vector<std::tuple<double, double, double, long long>> anchors;  // (x, y, σ, n)
};
struct MapOverlay {
    std::vector<MapOverlayRow> rows;
    std::optional<std::pair<double, double>> pos;   // (east, north) 已平移
    std::optional<double> heading;                  // rad
};

// 诊断快照 (既有板端库兼容字段集)
struct AttitudeDiag {
    bool enabled = false, has_data = false, matched = false,
         fresh = false, clamped = false;
    double applied_roll_deg = 0.0, applied_pitch_deg = 0.0;
    double raw_roll_deg = 0.0, raw_pitch_deg = 0.0;
    double age_s = -1.0, clamp_deg = 5.0, timeout_s = 0.5,
           match_window_s = 0.1;
};
struct RtkPosDiag {
    bool rtk_hint_valid = false;
    double axle_east_m = 0.0, axle_north_m = 0.0, heading_deg = 0.0;
    double rtk_diff_age_s = 0.0, rtk_speed_ms = 0.0;
    int rtk_pos_status = 0;
    bool matched = false, fresh = false;
    double match_age_s = -1.0, match_window_s = 0.1;
};
struct FusionState {
    int mode = 2;
    bool rtk_enu_active = false;
    double sigma_dist = 0.0, sigma_yaw = 0.0;
    int map_size = 0;
    bool engaged = true;
    int nav_mode = 1;
};

// ══════════════════════════════════════════════════════════════════
//  RowMapNavigation — 导航核心 (VisionNavigation 的接口兼容实现)
// ══════════════════════════════════════════════════════════════════
class RowMapNavigation {
public:
    RowMapNavigation(const GroundProjector::Config& proj_cfg,
                     const NavParams& prm = NavParams())
        : proj_(proj_cfg), prm_(prm), map_() {
        heading_var_ = prm_.sigma_heading_meas * prm_.sigma_heading_meas;
    }

    // ── 配置/控制接口 ──
    void set_debug(bool enabled) { debug_ = enabled; }
    void set_debug_verbose(bool) {}   // 兼容占位 (无 verbose 通道)

    void set_sensor_mode(int mode) { sensor_mode_ = mode; }

    // 仿真器/业务层喂上一帧控制模式 (扩展接口):
    // CAPTURE/掉头段解锁重捕获引导的行跨度限制 —— 捕获阶段车常在
    // 行带外, 查表回家是唯一生路; WORK 段维持限制 (防引导锁在行
    // 延长线上抑制行尾掉头语义)
    void set_control_hint(const std::string& mode) { ctrl_hint_ = mode; }

    // 仿真器/业务层喂车速 (扩展接口): 航向 EMA α 随速度自适应 ——
    // 高速时固定 α 的秒级滞后成倍放大, 反投影错位导致观测关联失败
    void set_speed_hint(double speed) {
        speed_hint_ = std::max(0.0, speed);
    }

    // RTK 天线杆臂 (兼容存储; P1 位姿直接采用天线位置, 杆臂进入
    // 常量系统偏移, 不影响相对导航量)
    void set_lever_arm(double fwd_m, double lat_m) {
        lever_fwd_m_ = fwd_m;
        lever_lat_m_ = lat_m;
    }
    void set_lookahead_time(double t_s) { lookahead_time_s_ = t_s; }

    // 导航输出使能 (false 时 get_navigation_data 返回全零)
    void set_nav_engaged(bool engaged) { nav_engaged_ = engaged; }

    // 行几何模式 (0=直线 1=曲线; 锚点折线天然支持曲线, 仅记录)
    bool set_nav_mode(int mode) {
        if (mode != 0 && mode != 1) return false;
        nav_mode_ = mode;
        return true;
    }

    // 最近一帧导航源模式 (兼容占位: 本实现无源切换, 恒 Unknown)
    int get_last_source_mode() const { return 0; }

    // RTK 速度可用性 (兼容: 固定解且速度超阈值)
    bool motion_speed_valid() const {
        return rtk_has_speed_ && pos_status_ == 4 && rtk_speed_ms_ > 0.05;
    }
    double motion_speed_ms() const { return rtk_speed_ms_; }

    // 行状态全量复位 (换行/重启用): 地图/锁定/位姿全部清空。
    // RTK 基准原点/航向 EMA/输出平滑状态若保留, 重播首帧会从旧
    // 状态爬升 —— 全部重建。
    void force_row_reset() {
        map_.reset();
        pos_.reset();
        heading_.reset();
        heading_var_ = prm_.sigma_heading_meas * prm_.sigma_heading_meas;
        origin_.reset();
        pos_status_ = 0;
        att_ = {0.0, 0.0};
        att_has_ = false;
        att_fresh_ = false;
        att_applied_last_ = false;
        frame_ = 0;
        out_valid_prev_ = false;
        quality_ = 0;
        nav_line_px_.clear();
        overlay_ = OverlayData{};
        world_origin_.reset();
        hd_span0_.clear();
        hd_span0_valid_ = false;
        row_alive_ = false;
        obs_hist_.clear();
        qual_hist_.clear();
    }

    // ── RTK/IMU 数据注入 ──
    // 位置以首包 lat/lon 为基准逆投影 ENU —— 地图坐标活在同一基准
    // 系, 相对量 (CTE/yaw) 不受基准偏移影响。航向 (双天线) EMA 平滑。
    // pos_status != 4 (非固定解) 时位置保持 (hint 失效语义)。
    void feed_rtk_data(const RtkPacket& p) {
        if (!std::isfinite(p.latitude) || !std::isfinite(p.longitude))
            return;
        if (!origin_) origin_ = std::make_pair(p.latitude, p.longitude);

        pos_status_ = p.pos_status;
        if (pos_status_ == 4) {
            const double lat0 = origin_->first, lon0 = origin_->second;
            const double cos_lat0 = std::cos(lat0 * kRadPerDeg);
            const double east = (p.longitude - lon0) * kMetresPerDeg * cos_lat0;
            const double north = (p.latitude - lat0) * kMetresPerDeg;
            pos_ = std::make_pair(east, north);
        }

        if (std::isfinite(p.heading_deg)) {
            const double h_rad = p.heading_deg * kRadPerDeg;
            if (!heading_) {
                heading_ = h_rad;
                heading_var_ = prm_.sigma_heading_meas * prm_.sigma_heading_meas;
            } else {
                // EMA (样本先 wrap 到当前估计 ±180°, 防绕圈跳变);
                // α 随车速自适应 (v=1 基准不变, 高速放大压滞后, 上限 0.6)
                const double dh = pymod(h_rad - *heading_ + kPi, 2.0 * kPi)
                                  - kPi;
                const double a = std::min(
                    0.6, prm_.heading_ema_alpha * std::max(1.0, speed_hint_));
                *heading_ += a * dh;
                // 先平方后乘 (固定结合序 `X*(σ*σ)`,
                // 左结合 `(X*σ)*σ` 会引入 ULP 舍入差)
                heading_var_ = (a / (2.0 - a))
                               * (prm_.sigma_heading_meas
                                      * prm_.sigma_heading_meas);
            }
        }

        if (std::isfinite(p.roll_deg) && std::isfinite(p.pitch_deg)) {
            att_ = {p.roll_deg * kRadPerDeg, p.pitch_deg * kRadPerDeg};
            att_has_ = true;
            att_fresh_ = true;
        }

        rtk_speed_ms_ = p.speed_ms;
        rtk_has_speed_ = true;
        rtk_diff_age_s_ = p.diff_age_s;
    }

    // ── 主入口: 每帧导航解算 ──
    NavOutput get_navigation_data(const std::vector<MaskView>& masks,
                                  double frame_time) {
        ++frame_;
        NavOutput out;
        if (!nav_engaged_) return out;   // 输出使能关闭 → 全零 (状态保持)

        // 1. 位姿可用性 (P1: RTK 主路径, mode 2)。
        // mode 0/1 无位置源 (视觉里程计位姿留 P2) → 诚实降级 q=0
        if (sensor_mode_ < 2 || !pos_ || !heading_) {
            att_fresh_ = false;
            quality_ = 0;
            return out;
        }
        const double px = pos_->first, py = pos_->second;
        const double c = std::cos(*heading_), s = std::sin(*heading_);

        // 2. 投影器姿态补偿 (mode>=1 且有 IMU 数据)
        const bool att_applied = (sensor_mode_ >= 1 && att_has_);
        if (att_applied)
            proj_.set_attitude(att_.first, att_.second);
        else
            proj_.set_attitude(0.0, 0.0);
        att_fresh_ = false;
        att_applied_last_ = att_applied;

        // 3. 观测提取: mask → 车辆系 → 世界系
        // C3 质量网络激活时: 逐 band 网络前向得 (δx 纠偏, σ²),
        // 替代手写三项式 var —— fuse() 接口不变, 地图核心零改动
        std::vector<std::vector<ObsPoint>> obs_rows;
        obs_rows.reserve(masks.size());
        for (const MaskView& m : masks) {
            std::vector<std::pair<double, double>> pts_v;
            std::vector<BandMeta> bands;
            extract_points(m, pts_v, bands);
            if (pts_v.size() < 2) continue;
            const std::vector<double>* var_list = nullptr;
            std::vector<double> vars_net;
            if (qnet_on_ && !bands.empty()) {
                // δx 安全限幅: 网络离谱输出不至污染观测
                vars_net.resize(bands.size());
                double feats[14], out2[2];
                for (std::size_t i = 0; i < bands.size(); ++i) {
                    band_features(bands, static_cast<int>(i), m.score,
                                  feats);
                    qnet_.forward(feats, out2);
                    const double dx = std::min(
                        std::max(out2[0], -prm_.qnet_max_dx),
                        prm_.qnet_max_dx);
                    pts_v[i].first += dx;
                    vars_net[i] = std::min(
                        std::max(std::exp(out2[1]), kQnetVarMin),
                        kQnetVarMax);
                }
                var_list = &vars_net;
            }
            std::vector<ObsPoint> pts_w;
            pts_w.reserve(pts_v.size());
            for (std::size_t i = 0; i < pts_v.size(); ++i) {
                const double xv = pts_v[i].first;
                const double zv = pts_v[i].second;
                const double wx = px + xv * c + zv * s;
                const double wy = py - xv * s + zv * c;
                const double var = var_list != nullptr
                    ? (*var_list)[i]
                    : ((prm_.k_pix_sigma * zv) * (prm_.k_pix_sigma * zv)
                       + heading_var_ * zv * zv
                       + prm_.sigma_rtk * prm_.sigma_rtk);
                pts_w.push_back(ObsPoint{wx, wy, var});
            }
            obs_rows.push_back(std::move(pts_w));
        }

        // 4. 地图更新 + 任务行锁定 (无观测帧跳过 GC, 地图保持)
        std::vector<int> matched = obs_rows.empty()
                                        ? std::vector<int>{}
                                        : map_.update(obs_rows, frame_);
        const std::optional<int> task_id = map_.lock_task_row(px, py);
        const bool task_obs =
            task_id.has_value()
            && std::find(matched.begin(), matched.end(), *task_id)
                   != matched.end();
        last_matched_task_ = task_obs;

        // 4.5 掉头中止检测: 任一已有行延续 = 掉头理由消失。
        // 掉头中 (ctrl_hint 非 WORK) 对岸锚点落图: 断垄时多行同时
        // 延伸, 但掉头段位姿漂移会让任务行的延续观测落进邻行 → 只
        // 盯任务行漏判。快照掉头开始时全部行的 y 跨度, 之后任一行
        // 增长 ≥2m 即行存活。掉头中新见的行 (新 id) 不在快照内不
        // 触发; 真地头所有行同止不触发。纯地图几何事实, 不依赖网络
        if (ctrl_hint_ != "WORK") {
            if (!hd_span0_valid_) {
                hd_span0_.clear();
                for (const auto& [rid, r] : map_.rows())
                    hd_span0_.emplace(
                        rid, std::make_pair(r.y_min(), r.y_max()));
                hd_span0_valid_ = true;
            } else {
                for (const auto& [rid, span0] : hd_span0_) {
                    const auto it = map_.rows().find(rid);
                    if (it == map_.rows().end()) continue;
                    if (it->second.y_max() > span0.second + 2.0
                            || it->second.y_min() < span0.first - 2.0) {
                        row_alive_ = true;
                        break;
                    }
                }
            }
        } else {
            hd_span0_.clear();
            hd_span0_valid_ = false;
            row_alive_ = false;
        }

        // 5. 解算: 任务行前段 → 车辆系 → 加权二次曲线拟合
        const RowModel* task = map_.task_row();
        std::optional<CurveFit> fit;
        std::vector<AnchorPoint> seg_world;
        if (task != nullptr) {
            // 前向段按航向的 +y 分量选段方向 (锚点按 y 有序): 反向
            // 再入 (掉头后沿 -y 开) 时前方是 y 较小侧, 固定 +y 段会
            // 取到身后锚点导致拟合错向
            if (std::cos(*heading_) >= 0.0) {
                seg_world = task->segment(py - prm_.seg_back_m,
                                          py + prm_.lookahead_m);
            } else {
                seg_world = task->segment(py - prm_.lookahead_m,
                                          py + prm_.seg_back_m);
            }
            if (static_cast<int>(seg_world.size())
                    >= prm_.min_solve_anchors) {
                std::vector<std::array<double, 3>> pts_v;
                pts_v.reserve(seg_world.size());
                for (const AnchorPoint& a : seg_world) {
                    const double dx = a.x - px, dy = a.y - py;
                    pts_v.push_back({dx * c - dy * s, dx * s + dy * c,
                                     a.var});
                }
                fit = fit_curve(pts_v);
            }
        }

        // 6. 输出与 quality 分级
        std::optional<CurveFit> guide;
        if (!fit) {
            // 前向段拟合失败分两种, 语义完全不同:
            // a) 车在行纵向跨度内但横向偏出/前向锚点不足 → 重捕获
            //    引导 (查表回家, 观测恢复后自动升级 95/70);
            // b) 车已越过行尾/行头 (>2m) → 引导反而有害: 会把车锁在
            //    行延长线上蛇形且 q=50 恒有效抑制行尾掉头。不引导,
            //    q=0 让控制器丢行计数 → 路程掉头语义接管
            if (task != nullptr) {
                const bool in_span =
                    (task->y_min() - 2.0 <= py) && (py <= task->y_max() + 2.0);
                // CAPTURE/掉头段解锁 span 限制: 捕获阶段车常在行带
                // 外, 最近锚点邻域引导是唯一生路; WORK 段维持限制
                if (in_span || ctrl_hint_ != "WORK") {
                    const std::vector<AnchorPoint> seg_g =
                        task->nearest_segment(py, prm_.recapture_anchors);
                    if (static_cast<int>(seg_g.size())
                            >= prm_.min_solve_anchors) {
                        std::vector<std::array<double, 3>> pts_g;
                        pts_g.reserve(seg_g.size());
                        for (const AnchorPoint& a : seg_g) {
                            const double dx = a.x - px, dy = a.y - py;
                            pts_g.push_back({dx * c - dy * s,
                                             dx * s + dy * c, a.var});
                        }
                        guide = fit_curve(pts_g);
                    }
                }
            }
        }

        if (!guide && !fit) {
            // 地图不足 (开局未建图 / 行尾外无锚点): hold 无效输出
            if (quality_ > 0 && debug_) {
                std::fprintf(stderr,
                             "[ROWMAP] quality->0 @ t=%.2fs pos=(%.2f,%.2f) "
                             "status=%d rows=%d task=%d task_anchors=%d "
                             "seg_len=%d obs_rows=%d\n",
                             frame_time, px, py, pos_status_,
                             static_cast<int>(map_.rows().size()),
                             map_.task_row_id().value_or(-1),
                             task ? static_cast<int>(task->size()) : 0,
                             static_cast<int>(seg_world.size()),
                             static_cast<int>(obs_rows.size()));
            }
            quality_ = 0;
            nav_line_px_.clear();
            overlay_ = OverlayData{};
            out_valid_prev_ = false;
            // C5 历史 + 概率前向 (q=0 帧正是行尾语义的关键样本)
            update_p_end(px, py, task_obs, false, false,
                         static_cast<int>(obs_rows.size()), 0);
            return out;
        }

        CurveFit fc{};
        int quality_out = 0;
        if (guide) {
            if (quality_ == 0 && debug_) {
                std::fprintf(stderr,
                             "[ROWMAP] recapture guide engaged @ t=%.2fs "
                             "pos=(%.2f,%.2f) task=%d\n",
                             frame_time, px, py,
                             map_.task_row_id().value_or(-1));
            }
            fc = *guide;
            quality_out = prm_.quality_recapture;
        } else {
            fc = *fit;
            quality_out = task_obs ? prm_.quality_tracking
                                   : prm_.quality_dead_reckon;
        }
        const double front_dist = -curve_x(fc, prm_.z_front);
        const double rear_dist = -curve_x(fc, prm_.z_rear);
        // yaw 取曲线在车体中心 (z=0) 处的切线方向 —— 行方向的一般化
        // (直线拟合时切线恒为 b, 与旧版语义一致)
        const double yaw_deg = -(std::atan(curve_dx(fc, 0.0)) * kDegPerRad);

        // 输出端单极点平滑: 压 RTK 噪声透传的输出跳动 (yaw/front)。
        // 无效→有效恢复首帧直通重置 (避免从 0 爬升拖慢恢复)
        if (out_valid_prev_) {
            const double ao = prm_.output_ema_alpha;
            yaw_s_ += ao * (yaw_deg - yaw_s_);
            front_s_ += ao * (front_dist - front_s_);
            rear_s_ += ao * (rear_dist - rear_s_);
        } else {
            yaw_s_ = yaw_deg;
            front_s_ = front_dist;
            rear_s_ = rear_dist;
            out_valid_prev_ = true;
        }

        // 导航线像素: 拟合曲线 z∈{0.5, 12} 端点投影 (与观测同一外参)
        nav_line_px_.clear();
        for (const double z_end : {0.5, 12.0}) {
            const auto [col, row] =
                proj_.vehicle_xz_to_pixel(curve_x(fc, z_end), z_end);
            if (0 <= col && col < proj_.crop_w() && 0 <= row
                && row < proj_.crop_h()) {
                nav_line_px_.emplace_back(col, row);
            }
        }

        // quality 分级: 95 跟踪 / 70 外推 / 50 重捕获引导
        quality_ = quality_out;

        // C5 历史 + 行尾概率前向 (可开关; 历史窗常驻维护)
        update_p_end(px, py, task_obs, fit.has_value(),
                     guide.has_value(),
                     static_cast<int>(obs_rows.size()), quality_out);

        // overlay: 各行锚点段 (车前方) 投影像素 + 是否任务行
        overlay_ = build_overlay(px, py, c, s);
        overlay_.valid = true;

        out.yaw_deg = yaw_s_;
        out.front_dist = front_s_;
        out.rear_dist = rear_s_;
        out.nav_line_px = nav_line_px_;
        out.quality = quality_;
        return out;
    }

    // ── 诊断 ──
    AttitudeDiag get_attitude_diag() const {
        AttitudeDiag d;
        const bool applied = att_applied_last_;
        d.enabled = sensor_mode_ >= 1;
        d.has_data = att_has_;
        d.matched = false;          // 本实现恒用最新值 (无时间匹配)
        d.fresh = applied;
        d.clamped = false;
        d.applied_roll_deg = applied ? att_.first * kDegPerRad : 0.0;
        d.applied_pitch_deg = applied ? att_.second * kDegPerRad : 0.0;
        d.raw_roll_deg = att_has_ ? att_.first * kDegPerRad
                                  : std::nan("");
        d.raw_pitch_deg = att_has_ ? att_.second * kDegPerRad
                                   : std::nan("");
        d.age_s = -1.0;
        d.clamp_deg = 5.0;
        d.timeout_s = 0.5;
        d.match_window_s = 0.1;
        return d;
    }

    RtkPosDiag get_rtk_position_diag() const {
        RtkPosDiag d;
        d.rtk_hint_valid = pos_.has_value() && pos_status_ == 4;
        d.axle_east_m = pos_ ? pos_->first : std::nan("");
        d.axle_north_m = pos_ ? pos_->second : std::nan("");
        d.heading_deg = heading_ ? *heading_ * kDegPerRad : std::nan("");
        d.rtk_diff_age_s = rtk_diff_age_s_;
        d.rtk_speed_ms = rtk_speed_ms_;
        d.rtk_pos_status = pos_status_;
        d.matched = pos_.has_value();
        d.fresh = d.rtk_hint_valid;
        d.match_age_s = -1.0;
        d.match_window_s = 0.1;
        return d;
    }

    FusionState get_fusion_state() const {
        FusionState f;
        f.mode = sensor_mode_;
        f.rtk_enu_active = pos_.has_value() && pos_status_ == 4;
        f.sigma_dist = std::nan("");
        f.sigma_yaw = std::nan("");
        f.map_size = map_.stats().total_anchors;
        f.engaged = nav_engaged_;
        f.nav_mode = nav_mode_;
        return f;
    }

    // ── 可视化 (GUI; 不进算法路径) ──
    const OverlayData& get_overlay_data() const { return overlay_; }

    // 俯视图地图可视化: 锚点含融合后 σ (=√var) 与观测次数 n。
    // 坐标系: 若已设 world_origin (首包真值起点) 则平移到田块系,
    // 否则保持 RTK 基准系 (退化显示, 原点=车辆首包位置)。
    MapOverlay get_map_overlay() const {
        const double ox = world_origin_ ? world_origin_->first : 0.0;
        const double oy = world_origin_ ? world_origin_->second : 0.0;
        MapOverlay mo;
        for (const auto& [rid, row] : map_.rows()) {
            if (row.size() < 2) continue;
            MapOverlayRow r;
            r.id = rid;
            r.task = map_.task_row_id().has_value()
                     && rid == *map_.task_row_id();
            r.anchors.reserve(row.size());
            for (const AnchorPoint& ap : row.anchors()) {
                r.anchors.emplace_back(ap.x + ox, ap.y + oy,
                                       std::sqrt(ap.var), ap.n);
            }
            mo.rows.push_back(std::move(r));
        }
        if (pos_) mo.pos = std::make_pair(pos_->first + ox,
                                          pos_->second + oy);
        mo.heading = heading_;
        return mo;
    }

    // RTK 基准系→田块系换算原点 (纯可视化用, 首包设置生效)
    void set_rtk_world_origin(double world_x, double world_y) {
        if (!world_origin_)
            world_origin_ = std::make_pair(world_x, world_y);
    }

    MapStats map_stats() const { return map_.stats(); }
    bool task_locked() const { return map_.task_row_id().has_value(); }
    bool last_matched_task() const { return last_matched_task_; }
    const MapParams& map_params() const { return map_.params(); }

    // 掉头中止信号 (扩展接口): 掉头期间任一已有行 y 跨度增长 ≥2m
    // (对岸延续锚点落图)。True = 行未结束, 控制器应中止掉头程序跳
    // 捕获段。WORK 帧恒 False
    bool get_row_alive() const { return row_alive_; }

    // ── 感知网络 (可开关, 消融 A/B 用; 学习只生产观测/概率,
    //    决策仍由本类几何核心做 —— 地图/解算接口零改动) ──

    // C3 异方差观测质量网络: 逐 band 前向得 (δx 纠偏, σ²) 替代
    // 手写三项式 var (δx 限幅 qnet_max_dx)。path 为空 → 恢复手写
    // 公式。返回加载是否成功 (维度不符/文件缺失 = false 且保持关闭)
    bool set_quality_net(const std::string& wts_path) {
        if (wts_path.empty()) {
            qnet_on_ = false;
            qnet_ = MlpNet{};
            return true;
        }
        MlpNet n;
        if (!n.load(wts_path) || n.in_dim != 14 || n.out_dim != 2) {
            qnet_on_ = false;
            return false;
        }
        qnet_ = std::move(n);
        qnet_on_ = true;
        return true;
    }

    // C5 行尾/地头感知网络: 每帧 (含 q=0 帧) 由地图历史特征前向
    // P(行终止), get_headland_prob() 取走喂控制器掉头触发门调制。
    // path 为空 → 恢复固定触发阈值 (概率恒 None)
    bool set_headland_net(const std::string& wts_path) {
        if (wts_path.empty()) {
            hnet_on_ = false;
            hnet_ = MlpNet{};
            return true;
        }
        MlpNet n;
        if (!n.load(wts_path) || n.in_dim != 22 || n.out_dim != 1) {
            hnet_on_ = false;
            return false;
        }
        hnet_ = std::move(n);
        hnet_on_ = true;
        return true;
    }

    // bytes 版 (pybind 绑定层用 Python I/O 读文件后转调): Windows
    // 中文系统 ANSI 代码页 (GBK) 下 C 窄字符 fopen 打不开 pybind 传
    // 入的 UTF-8 字节非 ASCII 路径, 绑定层读字节消解; n=0 → 关闭
    bool set_quality_net_bytes(const unsigned char* data, std::size_t n) {
        if (n == 0) {
            qnet_on_ = false;
            qnet_ = MlpNet{};
            return true;
        }
        MlpNet net;
        if (!net.load_buffer(data, n) || net.in_dim != 14
                || net.out_dim != 2) {
            qnet_on_ = false;
            return false;
        }
        qnet_ = std::move(net);
        qnet_on_ = true;
        return true;
    }

    bool set_headland_net_bytes(const unsigned char* data, std::size_t n) {
        if (n == 0) {
            hnet_on_ = false;
            hnet_ = MlpNet{};
            return true;
        }
        MlpNet net;
        if (!net.load_buffer(data, n) || net.in_dim != 22
                || net.out_dim != 1) {
            hnet_on_ = false;
            return false;
        }
        hnet_ = std::move(net);
        hnet_on_ = true;
        return true;
    }

    // C5 行尾概率 (未接入时 None; 每帧解算后更新)
    std::optional<double> get_headland_prob() const {
        if (!hnet_on_) return std::nullopt;
        return p_end_;
    }

private:
    // band 采样带元数据 (C3 特征组装用; 仅质量网络激活时收集)
    struct BandMeta {
        double row_c;      // band 像素行中心
        double span_px;    // band 内命中列跨度 (max−min)
        double xv, zv;     // 反投影车辆系坐标
        long ncols;        // band 内命中列数
    };

    // mask → 车辆系观测点串 (按图像行自底向上扫描)。
    // 每采样带取 mask 像素列均值作为行中心 (行条带横向连续),
    // 反投影到车辆系地面 (外参已含 IMU 姿态补偿 → 姿态残差进入
    // 观测噪声, 由 var 模型表达)。
    // bands 仅在质量网络激活时填充 (旁路全关时零额外开销), 与 pts
    // 逐元素平行 (同一 z 门限内通过的 band 才记录)
    void extract_points(const MaskView& m,
                        std::vector<std::pair<double, double>>& pts,
                        std::vector<BandMeta>& bands) const {
        const bool need_meta = qnet_on_;
        const int h = m.rows, w = m.cols;
        for (int r = h - 1; r >= 0; r -= prm_.scan_step_px) {
            const int lo = std::max(0, r - prm_.scan_step_px + 1);
            // band = rows [lo, r]: 每列是否出现过非零像素
            double col_sum = 0.0;
            long col_cnt = 0;
            int col_min = w, col_max = -1;
            for (int cc = 0; cc < w; ++cc) {
                bool any = false;
                for (int rr = lo; rr <= r; ++rr) {
                    if (m.data[static_cast<std::size_t>(rr) * w + cc]) {
                        any = true;
                        break;
                    }
                }
                if (any) {
                    col_sum += cc;
                    ++col_cnt;
                    if (cc < col_min) col_min = cc;
                    if (cc > col_max) col_max = cc;
                }
            }
            if (col_cnt > 0) {
                const double col = col_sum / static_cast<double>(col_cnt);
                const double row_c =
                    r - (prm_.scan_step_px - 1) / 2.0;
                const auto res = proj_.pixel_to_vehicle_xz(col, row_c);
                if (res && prm_.z_min_obs <= res->second
                    && res->second <= prm_.z_max_obs) {
                    pts.emplace_back(res->first, res->second);
                    if (need_meta) {
                        bands.push_back(BandMeta{
                            row_c,
                            static_cast<double>(col_max - col_min),
                            res->first, res->second, col_cnt});
                    }
                }
            }
        }
    }

    // 14 维 band 特征 (与训练侧 _band_features 严格同源; i 自底向上)
    void band_features(const std::vector<BandMeta>& bands, int i,
                       float score, double* f) const {
        const BandMeta& b = bands[static_cast<std::size_t>(i)];
        const int n = static_cast<int>(bands.size());
        // 相邻 band 横向跳变 (单 band 取 0): 条带断裂/错检指示
        double jump = 0.0;
        if (n > 1) {
            const int j = (i > 0) ? i - 1 : i + 1;
            jump = std::fabs(b.xv
                             - bands[static_cast<std::size_t>(j)].xv);
        }
        const double ch = proj_.crop_h(), cw = proj_.crop_w();
        f[0] = b.zv / prm_.z_max_obs;              // 前向距离
        f[1] = b.row_c / ch;                       // band 像素行中心
        f[2] = static_cast<double>(b.ncols) / cw;  // band 命中列数
        f[3] = b.span_px / cw;                     // 条带像素宽度
        f[4] = static_cast<double>(score);         // 分割置信度
        f[5] = att_applied_last_ ? att_.first * kDegPerRad : 0.0;
        f[6] = att_applied_last_ ? att_.second * kDegPerRad : 0.0;
        f[7] = std::sqrt(heading_var_);            // 航向 σ (rad)
        f[8] = jump;                               // 相邻 band 跳变 (m)
        f[9] = (i == 0) ? 1.0 : 0.0;               // 近端 band 指示
        f[10] = (i == n - 1) ? 1.0 : 0.0;          // 远端 band 指示
        f[11] = n / 100.0;                         // 本 mask band 总数
        f[12] = std::fabs(b.xv) / 3.0;             // 横向位置幅值 (m)
        f[13] = (pos_status_ == 4) ? 1.0 : 0.0;    // RTK 固定解
    }

    // 22 维行尾特征 (与训练侧 _headland_features 严格同源)。
    // 核心信号: 任务行锚点密度沿 y 的衰减 (真行尾平滑衰减到零且
    // 前方从未覆盖; 断垄前方曾有覆盖) + 行尾相对距离 + 观测命中/
    // quality 历史窗。无任务行时任务行相关槽位置零 + has_task=0
    std::array<double, 22> headland_features(
            double py, bool has_fit, bool has_guide,
            int n_obs_masks, int quality_now) const {
        std::array<double, 22> f{};
        // 近帧窗均值 (obs 窗 20 / quality 窗 8, 常驻维护)
        const auto obs_mean = [&](int last_n) {
            int cnt = 0;
            double s = 0.0;
            for (auto it = obs_hist_.rbegin();
                 it != obs_hist_.rend() && cnt < last_n;
                 ++it, ++cnt)
                s += *it ? 1.0 : 0.0;
            return cnt > 0 ? s / cnt : 0.0;
        };
        const double obs5 = obs_mean(5);
        const double obs15 = obs_mean(15);
        double qual_sum = 0.0;
        for (int q : qual_hist_) qual_sum += q;
        const double qual_mean = qual_hist_.empty()
            ? 0.0 : qual_sum / static_cast<double>(qual_hist_.size());

        const RowModel* task = map_.task_row();
        if (task != nullptr && !task->anchors().empty()) {
            const double y_min = task->y_min();
            const double y_max = task->y_max();
            long n05 = 0, n510 = 0, n1015 = 0, n1520 = 0;
            std::vector<double> front;
            front.reserve(task->size());
            for (const AnchorPoint& a : task->anchors()) {
                if (py <= a.y && a.y < py + 5.0) ++n05;
                else if (py + 5.0 <= a.y && a.y < py + 10.0) ++n510;
                else if (py + 10.0 <= a.y && a.y < py + 15.0) ++n1015;
                else if (py + 15.0 <= a.y && a.y < py + 20.0) ++n1520;
                if (py <= a.y && a.y <= py + 20.0) front.push_back(a.y);
            }
            // 前方 [py, py+20] 最大连续空窗 (含端点): 真行尾 → 锚点
            // 衰减后空窗占满; 断垄 → 远端有锚点压缩空窗。累加器从 0
            // 起步对相邻边差取 max (front 空时尾段 = 20 → 满窗)
            double gap_front = 0.0;
            double prev_edge = py;
            for (double y : front) {
                gap_front = std::max(gap_front, y - prev_edge);
                prev_edge = y;
            }
            gap_front = std::max(gap_front, py + 20.0 - prev_edge);
            const double gap_behind = std::max(0.0, py - y_max);
            f = {
                1.0,                             // 0 任务行存在
                std::log1p(static_cast<double>(task->size())),  // 1
                (y_max - y_min) / 100.0,         // 2 行总跨度
                (y_min - py) / 50.0,             // 3 行头相对距离
                (y_max - py) / 50.0,             // 4 行尾相对距离 (核心)
                std::log1p(static_cast<double>(n05)),   // 5 前方 0-5m
                std::log1p(static_cast<double>(n510)),  // 6
                std::log1p(static_cast<double>(n1015)), // 7
                std::log1p(static_cast<double>(n1520)), // 8
                gap_front / 20.0,                // 9 前方最大空窗
                gap_behind / 20.0,               // 10 车越过行尾距离
                obs5,                            // 11 近 5 帧命中率
                obs15,                           // 12 近 15 帧命中率
                qual_mean / 95.0,                // 13 quality 均值窗
                quality_now / 95.0,              // 14 本帧 quality
                std::log1p(static_cast<double>(n_obs_masks)),  // 15
                has_guide ? 1.0 : 0.0,           // 16 guide 状态
                has_fit ? 1.0 : 0.0,             // 17 fit 状态
                speed_hint_,                     // 18 车速
                std::sqrt(heading_var_),         // 19 航向 σ
                y_min / 100.0,                   // 20 行头绝对 y
                y_max / 100.0,                   // 21 行尾绝对 y
            };
            return f;
        }
        f = {
            0.0,                             // 0 无任务行
            0.0, 0.0, 0.0, 0.0,             // 1-4 任务行几何全零
            0.0, 0.0, 0.0, 0.0,             // 5-8 前方锚点数全零
            1.0,                             // 9 前方空窗 = 满
            0.0,                             // 10
            obs5, obs15, qual_mean / 95.0, quality_now / 95.0,
            std::log1p(static_cast<double>(n_obs_masks)),
            has_guide ? 1.0 : 0.0,
            has_fit ? 1.0 : 0.0,
            speed_hint_,
            std::sqrt(heading_var_),
            0.0, 0.0,                       // 20-21
        };
        return f;
    }

    // C5 历史窗推进 + 行尾概率前向 (每帧调用, 含 q=0 帧)。
    // 历史窗常驻维护 (是特征的一部分); 网络未接入时跳过特征组装
    void update_p_end(double px, double py, bool task_obs, bool has_fit,
                      bool has_guide, int n_obs_masks, int quality_now) {
        (void)px;
        obs_hist_.push_back(task_obs);
        if (obs_hist_.size() > 20) obs_hist_.pop_front();
        qual_hist_.push_back(quality_now);
        if (qual_hist_.size() > 8) qual_hist_.pop_front();
        if (!hnet_on_) return;
        const std::array<double, 22> feats = headland_features(
            py, has_fit, has_guide, n_obs_masks, quality_now);
        double out1[1];
        hnet_.forward(feats.data(), out1);
        p_end_ = 1.0 / (1.0 + std::exp(-out1[0]));
    }

    // ── 加权二次曲线拟合 ──
    // 结果曲线 x(z) = a + b·(z−zc) + c·(z−zc)² (zc = 加权中心化,
    // 数值条件良好); c≈0 时自动退化为直线 —— 直行/弯道统一表达。
    struct CurveFit {
        double a = 0.0, b = 0.0, c = 0.0, zc = 0.0;
    };

    static double curve_x(const CurveFit& f, double z) {
        const double u = z - f.zc;
        return f.a + f.b * u + f.c * u * u;
    }

    // 曲线切线斜率 dx/dz (z=0 处即车体中心下方行方向)
    static double curve_dx(const CurveFit& f, double z) {
        const double u = z - f.zc;
        return f.b + 2.0 * f.c * u;
    }

    // 车辆系点串加权拟合 (权重 1/var)。n≤3 或正规方程病态时退化为
    // 直线 (c=0); 点数不足返回无效。3×3 正规方程手写高斯消元,
    // 全程标量累加, 运算顺序确定。
    static std::optional<CurveFit> fit_curve(
            const std::vector<std::array<double, 3>>& pts) {
        const std::size_t n = pts.size();
        if (n < 2) return std::nullopt;
        // 加权矩 (标量累加)
        double sw = 0.0, wz = 0.0, wx = 0.0;
        for (const auto& p : pts) {
            const double w = 1.0 / std::max(p[2], 1e-8);
            sw += w;
            wz += w * p[1];
            wx += w * p[0];
        }
        const double zbar = wz / sw, xbar = wx / sw;

        // 直线回退 (n≤3 / 病态): x = a + b·z
        const auto linear = [&]() -> CurveFit {
            double denom = 0.0, num = 0.0;
            for (const auto& p : pts) {
                const double w = 1.0 / std::max(p[2], 1e-8);
                const double dz = p[1] - zbar;
                denom += w * dz * dz;
                num += w * dz * (p[0] - xbar);
            }
            if (std::fabs(denom) < 1e-9)
                return CurveFit{xbar, 0.0, 0.0, 0.0};
            const double b = num / denom;
            return CurveFit{xbar - b * zbar, b, 0.0, 0.0};
        };

        if (n <= 3) return linear();

        // 二次: 中心化矩 u = z − zbar → 正规方程 (加增广列)
        const double S = sw;
        double Su = 0.0, Suu = 0.0, Suuu = 0.0, Suuuu = 0.0;
        const double Sx = wx;
        double Sxu = 0.0, Sxuu = 0.0;
        for (const auto& p : pts) {
            const double w = 1.0 / std::max(p[2], 1e-8);
            const double u = p[1] - zbar;
            const double u2 = u * u;
            Su += w * u;
            Suu += w * u2;
            Suuu += w * u2 * u;
            Suuuu += w * u2 * u2;
            Sxu += w * u * p[0];
            Sxuu += w * u2 * p[0];
        }
        double m[3][4] = {{S, Su, Suu, Sx},
                          {Su, Suu, Suuu, Sxu},
                          {Suu, Suuu, Suuuu, Sxuu}};
        const double scale = std::max(std::max(std::fabs(S), std::fabs(Suu)),
                                      std::max(std::fabs(Suuu),
                                               std::fabs(Suuuu)));
        const double eps = 1e-12 * (scale > 1e-30 ? scale : 1e-30);
        // 高斯消元 (部分主元), 逐操作顺序确定
        for (int k = 0; k < 3; ++k) {
            int piv = k;
            for (int r = k + 1; r < 3; ++r)
                if (std::fabs(m[r][k]) > std::fabs(m[piv][k])) piv = r;
            if (std::fabs(m[piv][k]) < eps)
                return linear();          // 病态 (锚点 z 聚集) → 直线
            if (piv != k)
                for (int cc = 0; cc < 4; ++cc)
                    std::swap(m[piv][cc], m[k][cc]);
            for (int r = k + 1; r < 3; ++r) {
                const double f = m[r][k] / m[k][k];
                for (int cc = k; cc < 4; ++cc) m[r][cc] -= f * m[k][cc];
            }
        }
        const double c2 = m[2][3] / m[2][2];
        const double c1 = (m[1][3] - m[1][2] * c2) / m[1][1];
        const double c0 = (m[0][3] - m[0][2] * c2 - m[0][1] * c1) / m[0][0];
        return CurveFit{c0, c1, c2, zbar};
    }

    // 各行地图锚点段投影像素 (GUI 特征线叠层):
    // 每行 (锚点 >= min_display_anchors) 的像素点列表;
    // line_used = 该行是否任务行 —— 与基线 overlay 语义一致
    OverlayData build_overlay(double px, double py, double c, double s) {
        OverlayData ov;
        for (const auto& [rid, row] : map_.rows()) {
            if (static_cast<int>(row.size()) < prm_.min_display_anchors)
                continue;   // 噪声碎行不上图
            const std::vector<AnchorPoint> seg =
                row.segment(py - 0.5, py + prm_.z_max_obs);
            if (seg.size() < 2) continue;
            std::vector<std::pair<int, int>> px_list;
            px_list.reserve(seg.size());
            for (const AnchorPoint& ap : seg) {
                const double dx = ap.x - px, dy = ap.y - py;
                const double xv = dx * c - dy * s;
                const double zv = dx * s + dy * c;
                if (zv < 0.2) continue;   // 车后点不投影
                const auto [col, row_px] =
                    proj_.vehicle_xz_to_pixel(xv, zv);
                if (0 <= col && col < proj_.crop_w() && 0 <= row_px
                    && row_px < proj_.crop_h()) {
                    px_list.emplace_back(col, row_px);
                }
            }
            if (px_list.size() >= 2) {
                ov.center_lines.push_back(std::move(px_list));
                ov.line_used.push_back(
                    map_.task_row_id().has_value()
                    && rid == *map_.task_row_id());
            }
        }
        return ov;
    }

    // ── 状态 ──
    GroundProjector proj_;
    NavParams prm_;
    RowMapCore map_;

    std::optional<std::pair<double, double>> pos_;    // (east, north)
    std::optional<double> heading_;                   // rad 罗盘角 (EMA)
    double heading_var_ = 0.0;
    std::optional<std::pair<double, double>> origin_; // 首包 lat/lon 基准
    int pos_status_ = 0;

    std::pair<double, double> att_ = {0.0, 0.0};      // (roll, pitch) rad
    bool att_has_ = false;
    bool att_fresh_ = false;
    bool att_applied_last_ = false;

    long long frame_ = 0;
    int sensor_mode_ = 2;
    bool debug_ = false;

    bool out_valid_prev_ = false;
    double yaw_s_ = 0.0, front_s_ = 0.0, rear_s_ = 0.0;

    int quality_ = 0;
    std::vector<std::pair<int, int>> nav_line_px_;
    OverlayData overlay_;
    bool last_matched_task_ = false;

    std::optional<std::pair<double, double>> world_origin_;

    std::string ctrl_hint_ = "WORK";
    double speed_hint_ = 1.0;

    // ── 感知网络状态 (C3 质量网 / C5 行尾网, 可开关) ──
    MlpNet qnet_;
    bool qnet_on_ = false;
    MlpNet hnet_;
    bool hnet_on_ = false;
    double p_end_ = std::nan("");
    // C5 特征历史窗 (常驻维护, 是特征的一部分): 任务行观测命中 (20)
    // 与 quality (8); 未接入网络也推进, 接入后前几帧即有历史
    std::deque<bool> obs_hist_;
    std::deque<int> qual_hist_;

    // 掉头中止检测状态 (扩展接口): 掉头开始时全行 y 跨度快照
    // (rid → (y_min, y_max)) + 行存活标志。valid 标志区分
    // "快照为空地图" 与 "尚未快照"
    std::map<int, std::pair<double, double>> hd_span0_;
    bool hd_span0_valid_ = false;
    bool row_alive_ = false;

    bool nav_engaged_ = true;
    int nav_mode_ = 1;
    double lever_fwd_m_ = 0.0, lever_lat_m_ = 0.0;
    double lookahead_time_s_ = 0.0;
    double rtk_speed_ms_ = 0.0;
    double rtk_diff_age_s_ = 0.0;
    bool rtk_has_speed_ = false;
};

}  // namespace rowmap
