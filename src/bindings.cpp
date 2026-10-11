// ============================================================================
// src/bindings.cpp — pybind11 绑定层 (rowmap_navigation 工程)
//
// 模块名 vision_navigation: 与板端既有集成代码的导入路径保持二进制
// 兼容 (from navigation.vision_navigation import ...), 内部实现为本
// 工程独立的行结构地图 (RowMap) 算法, 与部署目标板端工程无
// 代码继承关系。
//
// 绑定面:
//   - RowMapCore / MapParams      (M1: 地图核心, 数值门禁用)
//   - MaskDataPlus                (mask 载体, 兼容鸭子类型输入)
//   - RowMapNavigation            (M2: 导航核心, VisionNavigation
//                                  接口兼容类)
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "rowmap/nav_core.hpp"
#include "rowmap/row_map.hpp"

#include <array>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;
using rowmap::AnchorPoint;
using rowmap::MapParams;
using rowmap::MaskView;
using rowmap::ObsPoint;
using rowmap::RowMapCore;

namespace {

// .wts 权重读为内存字节: 走 Python builtins.open (Unicode 路径安全 ——
// pybind str→std::string 得 UTF-8 字节, Windows 中文系统 ANSI/GBK
// 代码页的 C 窄字符 fopen 打不开非 ASCII 路径)。None/空串 → 空字节
// (网络关闭语义, 由 C++ setter 解释); 文件打不开 → nullopt (加载
// 失败, 与原生路径接口的返回 false 契约一致)
std::optional<std::string> read_wts_bytes(const py::object& arg) {
    if (arg.is_none()) return std::string{};
    if (py::isinstance<py::str>(arg)
            && arg.cast<std::string>().empty())
        return std::string{};
    try {
        py::object f = py::module_::import("builtins").attr("open")(
            arg, "rb");
        py::bytes data = f.attr("read")();
        f.attr("close")();
        return std::string(data);
    } catch (const py::error_already_set&) {
        return std::nullopt;
    }
}

// ── M1 逐项对照辅助 (RowMapCore) ─────────────────────────────────────────

std::vector<std::vector<ObsPoint>> obs_from_py(
        const std::vector<std::vector<std::tuple<double, double, double>>>&
            obs_rows) {
    std::vector<std::vector<ObsPoint>> out;
    out.reserve(obs_rows.size());
    for (const auto& row : obs_rows) {
        std::vector<ObsPoint> pts;
        pts.reserve(row.size());
        for (const auto& [wx, wy, var] : row) {
            pts.push_back(ObsPoint{wx, wy, var});
        }
        out.push_back(std::move(pts));
    }
    return out;
}

std::vector<std::tuple<double, double, double, long long>>
anchor_snapshot(const rowmap::RowModel& row) {
    std::vector<std::tuple<double, double, double, long long>> out;
    out.reserve(row.size());
    for (const AnchorPoint& a : row.anchors()) {
        out.emplace_back(a.y, a.x, a.var, a.n);
    }
    return out;
}

std::map<int,
         std::vector<std::tuple<double, double, double, long long>>>
rows_snapshot(const RowMapCore& self) {
    std::map<int,
             std::vector<std::tuple<double, double, double, long long>>>
        out;
    for (const auto& [rid, row] : self.rows()) {
        out.emplace(rid, anchor_snapshot(row));
    }
    return out;
}

py::dict map_stats_to_dict(const rowmap::MapStats& s) {
    py::dict d;
    d["n_rows"]        = s.n_rows;
    d["task_row_id"]   = s.task_row_id;
    d["task_anchors"]  = s.task_anchors;
    d["task_var_min"]  = s.task_var_min;
    d["total_anchors"] = s.total_anchors;
    return d;
}

// ── 相机参数 dict 解析 (构造函数用) ──────────────────────────────────
//
// 必需键: K_final (3x3), T_c2v (3x1/3), camera_rad (float)
// 可选键: crop_w / crop_h (缺省时 crop_h=image_height,
//         crop_w=2*cx 近似)。
// 基线库其余键 (R_c2v/R_v2c/T_v2c/vehicle_left_x/...) 存在则忽略
// —— 本实现外参由 camera_rad + T_c2v 按同式重建。

rowmap::Mat3 mat3_from_py(const py::object& obj, const char* name) {
    py::array_t<double, py::array::c_style | py::array::forcecast> arr =
        py::cast<py::array_t<double,
                             py::array::c_style | py::array::forcecast>>(
            obj);
    if (arr.ndim() != 2 || arr.shape(0) != 3 || arr.shape(1) != 3)
        throw std::runtime_error(
            std::string(name) + " must be a 3x3 array");
    auto buf = arr.unchecked<2>();
    rowmap::Mat3 m{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) m[i * 3 + j] = buf(i, j);
    return m;
}

rowmap::Vec3 vec3_from_py(const py::object& obj, const char* name) {
    py::array_t<double, py::array::c_style | py::array::forcecast> arr =
        py::cast<py::array_t<double,
                             py::array::c_style | py::array::forcecast>>(
            obj);
    const bool flat = (arr.ndim() == 1 && arr.shape(0) == 3);
    const bool col = (arr.ndim() == 2 && arr.shape(0) == 3
                      && arr.shape(1) == 1);
    if (!flat && !col)
        throw std::runtime_error(
            std::string(name) + " must be shape (3,) or (3,1)");
    auto buf = arr.mutable_unchecked();
    rowmap::Vec3 v{};
    for (int i = 0; i < 3; ++i)
        v[i] = flat ? buf(i) : buf(i, 0);
    return v;
}

double scalar_from_py(const py::dict& d, const char* key) {
    if (!d.contains(key))
        throw std::runtime_error(
            std::string("camera_info missing key: ") + key);
    return d[key].cast<double>();
}

// ── RTK dict → RtkPacket ─────────────────────────────────────────────

std::optional<double> opt_double(const py::dict& d, const char* key) {
    if (!d.contains(key)) return std::nullopt;
    const py::object v = d[key];
    if (v.is_none()) return std::nullopt;
    return v.cast<double>();
}

rowmap::RtkPacket rtk_from_pydict(const py::dict& d) {
    rowmap::RtkPacket p;
    const auto lat = opt_double(d, "latitude");
    const auto lon = opt_double(d, "longitude");
    // 必需字段缺失/None: 静默丢弃本包
    if (!lat || !lon) return p;
    p.latitude = *lat;
    p.longitude = *lon;
    p.altitude = opt_double(d, "altitude").value_or(0.0);
    p.speed_ms = opt_double(d, "speed_ms").value_or(0.0);
    p.speed_dir = opt_double(d, "speed_dir").value_or(0.0);
    p.recv_time = opt_double(d, "recv_time").value_or(0.0);
    if (d.contains("pos_status") && !d["pos_status"].is_none())
        p.pos_status = d["pos_status"].cast<int>();
    if (d.contains("sat_count") && !d["sat_count"].is_none())
        p.sat_count = d["sat_count"].cast<int>();
    p.diff_age_s = opt_double(d, "diff_age_s").value_or(0.0);
    p.roll_deg = opt_double(d, "roll").value_or(std::nan(""));
    p.pitch_deg = opt_double(d, "pitch").value_or(std::nan(""));
    p.heading_deg = opt_double(d, "heading").value_or(std::nan(""));
    return p;
}

// ── MaskDataPlus Python 载体 ─────────────────────────────────────────

struct MaskDataPlusPy {
    py::array mask;          // uint8 (H, W)
    int classic_class = 0;
    float score = 0.0f;
};

// 从鸭子类型对象提取 .mask (uint8 2D), 返回持有引用的连续缓冲
py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>
mask_array_from_obj(const py::object& item) {
    py::object mask_obj;
    if (py::isinstance<MaskDataPlusPy>(item)) {
        mask_obj = item.cast<MaskDataPlusPy&>().mask;
    } else {
        mask_obj = py::getattr(item, "mask");
    }
    return py::cast<py::array_t<std::uint8_t,
                                py::array::c_style
                                    | py::array::forcecast>>(mask_obj);
}

// ── overlay / diag → py 对象 ─────────────────────────────────────────

py::list line_list_to_py(
        const std::vector<std::pair<int, int>>& line) {
    py::list out;
    for (const auto& [c, r] : line) out.append(py::make_tuple(c, r));
    return out;
}

py::dict overlay_to_dict(const rowmap::OverlayData& ov) {
    py::dict d;
    if (!ov.valid) return d;
    py::list lines, used;
    for (const auto& line : ov.center_lines) {
        lines.append(line_list_to_py(line));
        used.append(py::bool_(false));  // 占位, 下面同步填
    }
    for (std::size_t i = 0; i < ov.line_used.size(); ++i)
        used[i] = py::bool_(ov.line_used[i]);
    d["center_lines"] = lines;
    d["line_used"] = used;
    return d;
}

py::dict map_overlay_to_dict(const rowmap::MapOverlay& mo) {
    py::dict d;
    py::list rows;
    for (const auto& r : mo.rows) {
        py::dict rd;
        rd["id"] = r.id;
        rd["task"] = r.task;
        py::list anchors;
        for (const auto& [x, y, sigma, n] : r.anchors)
            anchors.append(py::make_tuple(x, y, sigma, n));
        rd["anchors"] = anchors;
        rows.append(rd);
    }
    d["rows"] = rows;
    d["pos"] = mo.pos
                   ? py::make_tuple(mo.pos->first, mo.pos->second)
                   : py::none();
    d["heading"] = mo.heading ? py::float_(*mo.heading) : py::none();
    return d;
}

}  // namespace

PYBIND11_MODULE(vision_navigation, m) {
    m.doc() = "rowmap_navigation: row-structure-map visual navigation core";

    // ══ M1: 行结构地图 (数值门禁) ════════════════════════════════════
    py::class_<MapParams>(m, "MapParams")
        .def(py::init<>())
        .def_readwrite("anchor_step_m", &MapParams::anchor_step_m)
        .def_readwrite("anchor_fuse_m", &MapParams::anchor_fuse_m)
        .def_readwrite("gate_abs_m", &MapParams::gate_abs_m)
        .def_readwrite("task_lock_min_anchors",
                       &MapParams::task_lock_min_anchors)
        .def_readwrite("task_lock_tie_tol_m",
                       &MapParams::task_lock_tie_tol_m)
        .def_readwrite("stale_row_frames", &MapParams::stale_row_frames);

    py::class_<RowMapCore>(m, "RowMapCore")
        .def(py::init<MapParams>(), py::arg("params") = MapParams{})
        .def("reset", &RowMapCore::reset)
        .def("update",
             [](RowMapCore& self,
                const std::vector<std::vector<
                    std::tuple<double, double, double>>>& obs_rows,
                long long frame) {
                 return self.update(obs_from_py(obs_rows), frame);
             },
             py::arg("obs_rows"), py::arg("frame"),
             "观测点串列表 (每条 mask 一个, 点格式 (wx, wy, var)) "
             "关联/融合进地图, 返回本帧关联成功的行 id 列表")
        .def("lock_task_row", &RowMapCore::lock_task_row,
             py::arg("pose_x"), py::arg("pose_y"),
             "锁定任务行 (距车位最近的成熟行, 平局取 id 小)")
        .def("rows_snapshot", &rows_snapshot)
        .def("stats",
             [](const RowMapCore& self) {
                 return map_stats_to_dict(self.stats());
             })
        .def_property_readonly(
            "task_row_id",
            [](const RowMapCore& self) -> py::object {
                if (!self.task_row_id().has_value()) return py::none();
                return py::int_(*self.task_row_id());
            })
        .def_property_readonly(
            "params",
            [](const RowMapCore& self) { return self.params(); });

    // ══ MaskDataPlus (mask 载体) ═════════════════════════════════════
    py::class_<MaskDataPlusPy>(m, "MaskDataPlus")
        .def(py::init([](const py::array& mask, int classic_class,
                         float score) {
                 return MaskDataPlusPy{mask, classic_class, score};
             }),
             py::arg("mask"), py::arg("classic_class"), py::arg("score"))
        .def_readwrite("classic_class", &MaskDataPlusPy::classic_class)
        .def_readwrite("score", &MaskDataPlusPy::score)
        .def_readwrite("mask", &MaskDataPlusPy::mask);

    // ══ M2: 导航核心 (VisionNavigation 接口兼容) ════════════════════
    py::class_<rowmap::RowMapNavigation>(m, "VisionNavigation")
        .def(py::init([](double wheelbase_length, const py::dict& camera_info,
                         int image_height,
                         double vehicle_center_point_x_thresh,
                         bool test_mode) {
                 rowmap::GroundProjector::Config cfg;
                 if (!camera_info.contains("K_final")
                     || !camera_info.contains("T_c2v")
                     || !camera_info.contains("camera_rad"))
                     throw std::runtime_error(
                         "camera_info 必需键: K_final, T_c2v, camera_rad "
                         "(传入 vars(CameraInfo) 即可)");
                 cfg.K_final =
                     mat3_from_py(camera_info["K_final"], "K_final");
                 cfg.T_c2v = vec3_from_py(camera_info["T_c2v"], "T_c2v");
                 cfg.camera_rad =
                     scalar_from_py(camera_info, "camera_rad");
                 if (camera_info.contains("crop_w")
                     && !camera_info["crop_w"].is_none())
                     cfg.crop_w = camera_info["crop_w"].cast<int>();
                 else
                     cfg.crop_w = static_cast<int>(
                         2.0 * cfg.K_final[2]);
                 if (camera_info.contains("crop_h")
                     && !camera_info["crop_h"].is_none())
                     cfg.crop_h = camera_info["crop_h"].cast<int>();
                 else
                     cfg.crop_h = image_height;

                 rowmap::NavParams prm;
                 // 前后轴参考距离 = 半轴距 (按实际轴距自适应,
                 // 默认轴距 2.7m 下为 ±1.35)
                 prm.z_front = wheelbase_length / 2.0;
                 prm.z_rear = -wheelbase_length / 2.0;
                 return new rowmap::RowMapNavigation(cfg, prm);
             }),
             py::arg("wheelbase_length"), py::arg("camera_info"),
             py::arg("image_height"),
             py::arg("vehicle_center_point_x_thresh") = 0.2,
             py::arg("test_mode") = false)
        .def("set_debug", &rowmap::RowMapNavigation::set_debug,
             py::arg("enable") = true)
        .def("set_debug_verbose",
             &rowmap::RowMapNavigation::set_debug_verbose,
             py::arg("enable") = true)
        .def("set_sensor_mode", &rowmap::RowMapNavigation::set_sensor_mode,
             py::arg("mode"))
        .def("set_control_hint",
             &rowmap::RowMapNavigation::set_control_hint, py::arg("mode"))
        .def("set_speed_hint",
             &rowmap::RowMapNavigation::set_speed_hint, py::arg("speed"))
        .def("get_row_alive", &rowmap::RowMapNavigation::get_row_alive)
        .def("set_quality_net",
             [](rowmap::RowMapNavigation& self, py::object arg) {
                 const auto buf = read_wts_bytes(arg);
                 if (!buf) return false;
                 return self.set_quality_net_bytes(
                     reinterpret_cast<const unsigned char*>(buf->data()),
                     buf->size());
             },
             py::arg("wts_path_or_none"),
             "C3 质量网络 .wts 权重路径 (None/空=恢复手写 var); "
             "Returns: 加载是否成功")
        .def("set_headland_net",
             [](rowmap::RowMapNavigation& self, py::object arg) {
                 const auto buf = read_wts_bytes(arg);
                 if (!buf) return false;
                 return self.set_headland_net_bytes(
                     reinterpret_cast<const unsigned char*>(buf->data()),
                     buf->size());
             },
             py::arg("wts_path_or_none"),
             "C5 行尾网络 .wts 权重路径 (None/空=关闭, 概率恒 None); "
             "Returns: 加载是否成功")
        .def("get_headland_prob",
             [](const rowmap::RowMapNavigation& self) {
                 const auto p = self.get_headland_prob();
                 return p ? py::object(py::float_(*p))
                          : py::object(py::none());
             },
             "C5 行尾概率 P(end) (未接入时 None)")
        .def("set_lever_arm", &rowmap::RowMapNavigation::set_lever_arm,
             py::arg("fwd_m"), py::arg("lat_m"))
        .def("set_lookahead_time",
             &rowmap::RowMapNavigation::set_lookahead_time,
             py::arg("t_s"))
        .def("set_nav_engaged",
             &rowmap::RowMapNavigation::set_nav_engaged,
             py::arg("engaged"))
        .def("set_nav_mode", &rowmap::RowMapNavigation::set_nav_mode,
             py::arg("mode"))
        .def("get_last_source_mode",
             &rowmap::RowMapNavigation::get_last_source_mode)
        .def("motion_speed_valid",
             &rowmap::RowMapNavigation::motion_speed_valid)
        .def("motion_speed_ms",
             &rowmap::RowMapNavigation::motion_speed_ms)
        .def("force_row_reset",
             &rowmap::RowMapNavigation::force_row_reset)
        .def("feed_rtk_data",
             [](rowmap::RowMapNavigation& self, const py::dict& d) {
                 self.feed_rtk_data(rtk_from_pydict(d));
             },
             py::arg("rtk_data"))
        .def("get_navigation_data",
             [](rowmap::RowMapNavigation& self, py::object image,
                py::object plus_masks, double frame_time) {
                 // image 不参与解算 (mask-only 管线), 仅为接口兼容
                 std::vector<py::array_t<
                     std::uint8_t,
                     py::array::c_style | py::array::forcecast>>
                     holds;
                 std::vector<MaskView> views;
                 if (!plus_masks.is_none()) {
                     py::sequence seq =
                         plus_masks.cast<py::sequence>();
                     holds.reserve(seq.size());
                     views.reserve(seq.size());
                     for (py::handle item : seq) {
                         auto arr = mask_array_from_obj(item.cast<py::object>());
                         if (arr.ndim() != 2)
                             throw std::runtime_error(
                                 "mask must be 2D (H, W) uint8");
                         // 分割置信度 (鸭子类型: 无 score 属性缺省 1.0,
                         // 与 Python 侧 getattr(m, "score", 1.0) 一致)
                         float score = 1.0f;
                         try {
                             if (py::hasattr(item, "score"))
                                 score = item.attr("score").cast<float>();
                         } catch (const py::cast_error&) {
                             score = 1.0f;
                         }
                         holds.push_back(std::move(arr));
                         auto& a = holds.back();
                         views.push_back(MaskView{
                             a.data(),
                             static_cast<int>(a.shape(0)),
                             static_cast<int>(a.shape(1)),
                             score});
                     }
                 }
                 const rowmap::NavOutput out =
                     self.get_navigation_data(views, frame_time);
                 return py::make_tuple(
                     out.yaw_deg, out.front_dist, out.rear_dist,
                     line_list_to_py(out.nav_line_px), out.quality);
             },
             py::arg("image"), py::arg("plus_masks") = py::none(),
             py::arg("frame_time") = 0.0,
             "每帧导航解算: Returns (yaw_deg, front_dist, rear_dist, "
             "nav_line_px, quality)")
        .def("get_attitude_diag",
             [](const rowmap::RowMapNavigation& self) {
                 const auto d = self.get_attitude_diag();
                 py::dict out;
                 out["enabled"] = d.enabled;
                 out["has_data"] = d.has_data;
                 out["matched"] = d.matched;
                 out["fresh"] = d.fresh;
                 out["clamped"] = d.clamped;
                 out["applied_roll_deg"] = d.applied_roll_deg;
                 out["applied_pitch_deg"] = d.applied_pitch_deg;
                 out["raw_roll_deg"] = d.raw_roll_deg;
                 out["raw_pitch_deg"] = d.raw_pitch_deg;
                 out["age_s"] = d.age_s;
                 out["clamp_deg"] = d.clamp_deg;
                 out["timeout_s"] = d.timeout_s;
                 out["match_window_s"] = d.match_window_s;
                 return out;
             })
        .def("get_rtk_position_diag",
             [](const rowmap::RowMapNavigation& self) {
                 const auto d = self.get_rtk_position_diag();
                 py::dict out;
                 out["rtk_hint_valid"] = d.rtk_hint_valid;
                 out["axle_east_m"] = d.axle_east_m;
                 out["axle_north_m"] = d.axle_north_m;
                 out["heading_deg"] = d.heading_deg;
                 out["rtk_diff_age_s"] = d.rtk_diff_age_s;
                 out["rtk_speed_ms"] = d.rtk_speed_ms;
                 out["rtk_pos_status"] = d.rtk_pos_status;
                 out["matched"] = d.matched;
                 out["fresh"] = d.fresh;
                 out["match_age_s"] = d.match_age_s;
                 out["match_window_s"] = d.match_window_s;
                 return out;
             })
        .def("get_fusion_state",
             [](const rowmap::RowMapNavigation& self) {
                 const auto f = self.get_fusion_state();
                 py::dict out;
                 out["mode"] = f.mode;
                 out["rtk_enu_active"] = f.rtk_enu_active;
                 out["sigma_dist"] = f.sigma_dist;
                 out["sigma_yaw"] = f.sigma_yaw;
                 out["map_size"] = f.map_size;
                 out["engaged"] = f.engaged;
                 out["nav_mode"] = f.nav_mode;
                 return out;
             })
        .def("get_overlay_data",
             [](const rowmap::RowMapNavigation& self) {
                 return overlay_to_dict(self.get_overlay_data());
             })
        .def("get_map_overlay",
             [](const rowmap::RowMapNavigation& self) {
                 return map_overlay_to_dict(self.get_map_overlay());
             })
        .def("set_rtk_world_origin",
             &rowmap::RowMapNavigation::set_rtk_world_origin,
             py::arg("world_x"), py::arg("world_y"))
        .def("map_stats",
             [](const rowmap::RowMapNavigation& self) {
                 return map_stats_to_dict(self.map_stats());
             })
        .def_property_readonly(
            "task_locked",
            &rowmap::RowMapNavigation::task_locked);
}
