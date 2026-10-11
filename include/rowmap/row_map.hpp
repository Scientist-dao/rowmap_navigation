// ============================================================================
// rowmap/row_map.hpp — 行结构地图核心数据结构 (RowMap C++ 版, M1)
//
// 本工程为独立的视觉导航核心算法实现 (rowmap_navigation), 与既有
// 与部署目标板端工程无代码继承关系; 行结构地图、1/var 信息加权
// 融合、任务行 ID 锁定等算法设计均为本工程独立完成。
//
// 设计要点:
//   - 锚点折线表达: 每条作物行是世界系下沿行方向 (y) 有序的锚点串,
//     每个锚点持有横向位置 x 与方差 var; 行天然支持直线/弯道任意形状。
//   - 1/var 信息加权融合: 新观测与既有锚点按信息权重合成, 方差随
//     观测次数单调下降 —— 同一锚点看得越多越准。
//   - 行级关联: 每帧每条 mask 的观测点串按平均横向偏差与既有行匹配
//     (最近者且低于门限), 未匹配则建新行; 门限小于行距, 邻行不会误配。
//   - 任务行 ID 锁定: 一经锁定终身不变, 导航解算只依赖锁定行的地图
//     位置, 从机理上免疫车正下方对称中心的死锁位形。
//
// 坐标系: 世界系 (ENU), 行沿 y 延伸, 行间沿 x 排列。
//
// 数值约定:
//   - 全部双精度, 运算顺序确定, 不引入额外代数变形;
//   - x_at 端外用端段斜率外推 (var 同段插值), 供任务行锁定;
//     关联打分 mean_abs_dx 端外钳位取端锚点 x (农行直行先验);
//   - 方差下限 1e-8 (防除零), 空行查询返回 (0, 1e6)。
// ============================================================================

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace rowmap {

// ── 地图参数 (集中常量, 调参入口) ──
struct MapParams {
    double anchor_step_m      = 0.5;   // 新建行的锚点最小间距 (m, 沿行方向)
    double anchor_fuse_m      = 0.30;  // 观测 y 距最近锚点小于此值则融合
    double gate_abs_m         = 0.35;  // 行级关联门限: 平均横向偏差 (m)
    int    task_lock_min_anchors = 2;  // 成为任务行候选的最少锚点数
    double task_lock_tie_tol_m = 0.05; // 锁定平局容差 (m), 平局取 id 小
    int    stale_row_frames   = 30;    // 垃圾行回收: 无观测帧数阈值
    int    gc_min_anchors     = 4;     // 垃圾行回收: 锚点数下限 (少于此才回收)
};

// ── 基础数值类型 ──

// 行锚点: 行方向位置 y 处的横向位置估计
// y 固定取首次观测值 (y 误差只影响插值位置, 不融合);
// x 按 1/var 信息权重融合, 融合后 var = 1/(w_old+w_new) 单调下降。
struct AnchorPoint {
    double y   = 0.0;
    double x   = 0.0;
    double var = 1e-8;
    long long n = 1;

    AnchorPoint() = default;
    AnchorPoint(double y_, double x_, double var_)
        : y(y_), x(x_), var(std::max(var_, 1e-8)), n(1) {}

    // 信息加权融合一个观测: σ 融合 ~ σ_obs/√n 收敛
    void fuse(double x_obs, double var_obs) {
        const double w_old = 1.0 / var;
        const double w_new = 1.0 / std::max(var_obs, 1e-8);
        x   = (x * w_old + x_obs * w_new) / (w_old + w_new);
        var = 1.0 / (w_old + w_new);
        ++n;
    }
};

// 观测点: mask 反投影到世界系的一个点 (wx, wy) 及其方差
struct ObsPoint {
    double x   = 0.0;   // 世界系横向 (m)
    double y   = 0.0;   // 世界系纵向 (m, 行方向)
    double var = 1e-8;  // 横向方差 (m^2)
};

// 地图统计快照 (诊断/GUI)
struct MapStats {
    int n_rows         = 0;
    long long task_row_id = -1;   // -1 = 未锁定
    int task_anchors   = 0;
    double task_var_min = std::numeric_limits<double>::quiet_NaN();
    int total_anchors  = 0;
};

// ── 单条作物行: 世界系锚点折线 (按 y 升序维护) ──
class RowModel {
public:
    RowModel() = default;  // std::map 容器需要
    explicit RowModel(int row_id) : row_id_(row_id) {}

    int row_id() const { return row_id_; }
    long long last_seen() const { return last_seen_; }
    const std::vector<AnchorPoint>& anchors() const { return anchors_; }
    bool empty() const { return anchors_.empty(); }
    std::size_t size() const { return anchors_.size(); }

    double y_min() const {
        return anchors_.empty()
            ? std::numeric_limits<double>::quiet_NaN() : anchors_.front().y;
    }
    double y_max() const {
        return anchors_.empty()
            ? std::numeric_limits<double>::quiet_NaN() : anchors_.back().y;
    }

    // 行在 y 处的 (x, var): 段内线性插值, 端外用端段斜率外推
    std::pair<double, double> x_at(double y) const {
        const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(anchors_.size());
        if (n == 0) return {0.0, 1e6};
        if (n == 1) {
            const AnchorPoint& a = anchors_[0];
            return {a.x, a.var};
        }
        // 首个 y_i >= y 的位置, 夹到 [1, n-1] 得所在/最近段的右锚点
        std::ptrdiff_t i = std::lower_bound(
            anchors_.begin(), anchors_.end(), y,
            [](const AnchorPoint& a, double v) { return a.y < v; })
            - anchors_.begin();
        i = std::min(std::max(i, std::ptrdiff_t(1)), n - 1);
        const AnchorPoint& a0 = anchors_[i - 1];
        const AnchorPoint& a1 = anchors_[i];
        const double dy = a1.y - a0.y;
        const double t  = std::fabs(dy) > 1e-9 ? (y - a0.y) / dy : 1.0;
        const double x   = a0.x + t * (a1.x - a0.x);
        const double var = a0.var + std::min(std::max(t, 0.0), 1.0)
                           * (a1.var - a0.var);
        return {x, std::max(var, 1e-8)};
    }

    // 观测点串对本行的平均横向偏差 (行级关联打分); 空行返回 +inf。
    // 端外查询钳位取端锚点 x 而非斜率外推: 端段基线短, 斜率噪声
    // 长距外推会放大成关联门限量级的横向位移, 把行预测线拽到
    // 邻行位置劫持关联 (断垄对岸首帧跟错行的根因)。
    double mean_abs_dx(const std::vector<ObsPoint>& pts) const {
        if (anchors_.empty())
            return std::numeric_limits<double>::infinity();
        double total = 0.0;
        for (const ObsPoint& p : pts) {
            total += std::fabs(p.x - x_at_seq(anchors_, p.y));
        }
        return total / static_cast<double>(pts.size());
    }

    // [y0, y1] 内的锚点段 (供解算/可视化)
    std::vector<AnchorPoint> segment(double y0, double y1) const {
        auto first = std::lower_bound(
            anchors_.begin(), anchors_.end(), y0,
            [](const AnchorPoint& a, double v) { return a.y < v; });
        auto last = std::upper_bound(
            anchors_.begin(), anchors_.end(), y1,
            [](double v, const AnchorPoint& a) { return v < a.y; });
        return std::vector<AnchorPoint>(first, last);
    }

    // 离纵向位置 y 最近的 n 个锚点 (重捕获引导用):
    // 车可能远离行带, 以车为中心的窗会与锚点区完全错开,
    // 取最近锚点邻域保证查表总有解 (引导 = 朝最近已建段归位)。
    std::vector<AnchorPoint> nearest_segment(double y, int n = 8) const {
        if (anchors_.empty() || n <= 0) return {};
        const std::ptrdiff_t len =
            static_cast<std::ptrdiff_t>(anchors_.size());
        std::ptrdiff_t i = std::lower_bound(
            anchors_.begin(), anchors_.end(), y,
            [](const AnchorPoint& a, double v) { return a.y < v; })
            - anchors_.begin();
        std::ptrdiff_t lo = std::max<std::ptrdiff_t>(0, i - n / 2);
        std::ptrdiff_t hi = std::min<std::ptrdiff_t>(
            len, lo + static_cast<std::ptrdiff_t>(n));
        lo = std::max<std::ptrdiff_t>(0, hi - static_cast<std::ptrdiff_t>(n));
        return std::vector<AnchorPoint>(
            anchors_.begin() + lo, anchors_.begin() + hi);
    }

    // 观测点串 (按 y 升序) 融合进锚点串: 每点找 y 最近锚点,
    // 距离 < anchor_fuse_m 融合, 否则按 y 有序插入新锚点 ——
    // 锚点自然间隔 >= anchor_fuse_m, 密度有界。
    void absorb(const std::vector<ObsPoint>& pts, long long frame,
                const MapParams& prm) {
        last_seen_ = frame;
        for (const ObsPoint& p : pts) {
            const std::ptrdiff_t j = std::lower_bound(
                anchors_.begin(), anchors_.end(), p.y,
                [](const AnchorPoint& a, double v) { return a.y < v; })
                - anchors_.begin();
            std::ptrdiff_t best_j = -1;
            double best_d = prm.anchor_fuse_m;
            for (std::ptrdiff_t jj = j - 1; jj <= j; ++jj) {
                if (jj < 0 || jj >= static_cast<std::ptrdiff_t>(anchors_.size()))
                    continue;
                const double d = std::fabs(anchors_[jj].y - p.y);
                if (d < best_d) { best_d = d; best_j = jj; }
            }
            if (best_j >= 0) {
                anchors_[best_j].fuse(p.x, p.var);
            } else {
                anchors_.insert(
                    anchors_.begin() + j, AnchorPoint(p.y, p.x, p.var));
            }
        }
    }

    // 从观测点串 (按 y 升序) 新建行, 按 anchor_step_m 重采样建锚点
    static RowModel new_from_points(int row_id,
                                    const std::vector<ObsPoint>& pts,
                                    long long frame, const MapParams& prm) {
        RowModel row(row_id);
        row.last_seen_ = frame;
        double next_y = -std::numeric_limits<double>::infinity();
        for (const ObsPoint& p : pts) {
            if (p.y >= next_y) {
                row.anchors_.emplace_back(p.y, p.x, p.var);
                next_y = p.y + prm.anchor_step_m;
            }
        }
        return row;
    }

private:
    // 锚点序列 (按 y 升序) 在 y 处的 x: 段内线性插值, 端外钳位
    // 取端锚点 x (农行直行先验)。关联打分专用, 与 x_at 的端段
    // 斜率外推分工。
    static double x_at_seq(const std::vector<AnchorPoint>& anchors,
                           double y) {
        const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(anchors.size());
        if (n == 1) return anchors[0].x;
        std::ptrdiff_t i = std::lower_bound(
            anchors.begin(), anchors.end(), y,
            [](const AnchorPoint& a, double v) { return a.y < v; })
            - anchors.begin();
        i = std::min(std::max(i, std::ptrdiff_t(1)), n - 1);
        const AnchorPoint& a0 = anchors[i - 1];
        const AnchorPoint& a1 = anchors[i];
        const double dy = a1.y - a0.y;
        double t = std::fabs(dy) > 1e-9 ? (y - a0.y) / dy : 1.0;
        t = std::min(std::max(t, 0.0), 1.0);
        return a0.x + t * (a1.x - a0.x);
    }

    int row_id_ = 0;
    std::vector<AnchorPoint> anchors_;  // 按 y 升序 (不变量)
    long long last_seen_ = -1;
};

// ── 行结构地图: 多行管理 + 行级关联 + 任务行 ID 锁定 ──
class RowMapCore {
public:
    explicit RowMapCore(MapParams prm = MapParams()) : prm_(prm) {}

    void reset() {
        rows_.clear();
        next_id_ = 0;
        task_row_id_.reset();
    }

    const MapParams& params() const { return prm_; }

    // 观测点串列表 (每条 mask 一个) 关联/融合进地图。
    // 返回本帧关联成功的行 id 列表 (输入顺序), 供质量判定。
    std::vector<int> update(
            const std::vector<std::vector<ObsPoint>>& obs_rows,
            long long frame) {
        std::vector<int> matched;
        matched.reserve(obs_rows.size());
        for (const auto& pts_raw : obs_rows) {
            // 按 y 升序 (稳定排序, 同 y 观测顺序确定)
            std::vector<ObsPoint> pts = pts_raw;
            std::stable_sort(pts.begin(), pts.end(),
                             [](const ObsPoint& a, const ObsPoint& b) {
                                 return a.y < b.y;
                             });
            if (pts.size() < 2) continue;
            // 行级关联: 平均横向偏差最小且低于门限 (平局取先建行,
            // std::map 按 id 升序遍历, id 即建行先后序)
            int best_id = -1;
            double best_cost = prm_.gate_abs_m;
            for (const auto& [rid, row] : rows_) {
                const double cost = row.mean_abs_dx(pts);
                if (cost < best_cost) { best_cost = cost; best_id = rid; }
            }
            if (best_id >= 0) {
                rows_[best_id].absorb(pts, frame, prm_);
                matched.push_back(best_id);
            } else {
                const int new_id = next_id_++;
                rows_.emplace(new_id, RowModel::new_from_points(
                    new_id, pts, frame, prm_));
                matched.push_back(new_id);
            }
        }
        gc_stale_rows(frame);
        return matched;
    }

    // 任务行锁定: 距当前车位最近的成熟行; 平局 (距离差 < 容差)
    // 取 id 小 (先建 = 画面左侧行 = 骑行中心行语义)。
    // 锁定后 ID 身份固定 —— 死锁免疫的来源。
    std::optional<int> lock_task_row(double pose_x, double pose_y) {
        if (task_row_id_.has_value()) return task_row_id_;
        std::vector<std::pair<double, int>> cands;
        for (const auto& [rid, row] : rows_) {
            if (static_cast<int>(row.size()) < prm_.task_lock_min_anchors)
                continue;
            cands.emplace_back(
                std::fabs(row.x_at(pose_y).first - pose_x), rid);
        }
        if (cands.empty()) return std::nullopt;
        double d_min = std::numeric_limits<double>::infinity();
        for (const auto& [d, rid] : cands) d_min = std::min(d_min, d);
        int chosen = std::numeric_limits<int>::max();
        for (const auto& [d, rid] : cands) {
            if (d - d_min < prm_.task_lock_tie_tol_m)
                chosen = std::min(chosen, rid);
        }
        task_row_id_ = chosen;
        return task_row_id_;
    }

    const RowModel* task_row() const {
        if (!task_row_id_.has_value()) return nullptr;
        auto it = rows_.find(*task_row_id_);
        return it == rows_.end() ? nullptr : &it->second;
    }

    RowModel* task_row() {
        if (!task_row_id_.has_value()) return nullptr;
        auto it = rows_.find(*task_row_id_);
        return it == rows_.end() ? nullptr : &it->second;
    }

    const std::map<int, RowModel>& rows() const { return rows_; }
    const std::optional<int>& task_row_id() const { return task_row_id_; }

    MapStats stats() const {
        MapStats s;
        s.n_rows = static_cast<int>(rows_.size());
        s.task_row_id = task_row_id_.value_or(-1);
        int total = 0;
        for (const auto& [rid, row] : rows_) {
            total += static_cast<int>(row.size());
            if (task_row_id_.has_value() && rid == *task_row_id_) {
                s.task_anchors = static_cast<int>(row.size());
                if (!row.empty()) {
                    double vmin = row.anchors().front().var;
                    for (const AnchorPoint& a : row.anchors())
                        vmin = std::min(vmin, a.var);
                    s.task_var_min = vmin;
                }
            }
        }
        s.total_anchors = total;
        return s;
    }

private:
    // 回收垃圾行: 长期无观测且锚点过少的非任务行 (噪声 mask 残留)。
    // 任务行豁免 —— 锁定身份不得被回收破坏。
    void gc_stale_rows(long long frame) {
        std::vector<int> stale;
        for (const auto& [rid, row] : rows_) {
            const bool is_task = task_row_id_.has_value()
                                 && rid == *task_row_id_;
            if (!is_task
                && static_cast<int>(row.size()) < prm_.gc_min_anchors
                && frame - row.last_seen() > prm_.stale_row_frames) {
                stale.push_back(rid);
            }
        }
        for (int rid : stale) rows_.erase(rid);
    }

    MapParams prm_;
    std::map<int, RowModel> rows_;   // id 升序遍历 (id 单调递增分配)
    int next_id_ = 0;
    std::optional<int> task_row_id_;
};

}  // namespace rowmap
