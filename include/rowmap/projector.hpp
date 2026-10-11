// ============================================================================
// rowmap/projector.hpp — 地面投影器 (RowMap C++ 版, M2)
//
// 车辆系地面点 (X, 0, Z) ⇄ 裁剪图像像素 (col, row) 的双向投影,
// 含车身姿态 (roll/pitch) 补偿下的等效外参重建。
//
// 姿态补偿与投影语义:
//   - set_attitude: pitch 修改有效俯角重建 R_c2v, roll 经 R_unlevel
//     把倾斜车身系恢复到水平地面系, 组合后更新 R_v2c/T_v2c;
//   - 前向投影: 车辆系地面点经 R_v2c/T_v2c 与内参 K 投影到裁剪
//     图像像素;
//   - 像素取整用 round-half-even (banker's rounding);
//   - 反投影: d = K⁻¹·p 的三角回代, 射线与车辆系 Y=0 地平面求交,
//     相机后方/近平行返回无效。
//
// 仅依赖标准库 (3×3 矩阵手写), 无 OpenCV —— 板端交叉编译零负担。
// ============================================================================

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

namespace rowmap {

// ── 小型线性代数 (行主序 3×3 / 3×1) ──
using Mat3 = std::array<double, 9>;   // m[r*3+c]
using Vec3 = std::array<double, 3>;

inline Vec3 mat3_mul_vec3(const Mat3& m, const Vec3& v) {
    return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}

inline Mat3 mat3_mul_mat3(const Mat3& a, const Mat3& b) {
    Mat3 r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += a[i * 3 + k] * b[k * 3 + j];
            r[i * 3 + j] = s;
        }
    return r;
}

inline Mat3 mat3_transpose(const Mat3& m) {
    return {m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]};
}

// banker's rounding (half to even)
inline double round_half_even(double v) {
    const double fl = std::floor(v);
    const double diff = v - fl;
    if (diff > 0.5) return fl + 1.0;
    if (diff < 0.5) return fl;
    return (std::fmod(fl, 2.0) == 0.0) ? fl : fl + 1.0;
}

// ── 地面投影器 ──
class GroundProjector {
public:
    struct Config {
        Mat3 K_final{};         // 裁剪/缩放后的内参
        Vec3 T_c2v{};           // 相机系→车辆系平移 (nominal)
        double camera_rad = 0.0; // 相机名义俯角 (rad, 向下为正)
        int crop_w = 0;         // 裁剪图像宽 (px)
        int crop_h = 0;         // 裁剪图像高 (px)
    };

    explicit GroundProjector(const Config& cfg) : cfg_(cfg) {
        reset_extrinsics();
    }

    int crop_w() const { return cfg_.crop_w; }
    int crop_h() const { return cfg_.crop_h; }
    double camera_rad() const { return cfg_.camera_rad; }
    std::pair<double, double> attitude_rad() const { return {roll_, pitch_}; }

    // 更新车身姿态, 重建相机等效外参
    //
    // 相机刚接在车身上, 车身倾斜时相机随动:
    //   - pitch (抬头正): 有效俯角 = nominal - pitch, 重建 R_c2v;
    //   - roll (右侧下沉正): R_unlevel 把倾斜车身系恢复到水平地面系。
    // 传 (0, 0) 恢复 nominal 外参。
    void set_attitude(double roll_rad, double pitch_rad) {
        roll_ = roll_rad;
        pitch_ = pitch_rad;

        // 1. Pitch: 有效俯角下的 R_c2v
        const double camera_rad_eff = cfg_.camera_rad - pitch_rad;
        const double c = std::cos(camera_rad_eff), s = std::sin(camera_rad_eff);
        const Mat3 R_c2v_pitch = {1.0, 0.0, 0.0,
                                  0.0,  c,   s,
                                  0.0, -s,   c};

        // 2. Roll: unlevel 旋转 (倾斜车身系 -> 水平系, 绕 Z 前向轴)
        const double cr = std::cos(roll_rad), sr = std::sin(roll_rad);
        const Mat3 R_unlevel = {cr, -sr, 0.0,
                                sr,  cr, 0.0,
                                0.0, 0.0, 1.0};

        // 3. 组合有效外参: R_c2v_eff = R_unlevel @ R_c2v_pitch
        //    v_level = R_unlevel @ (R_c2v @ p_c + T_c2v)
        //    投影方向: p_c = R_c2v_eff^T @ (v_level - T_c2v_eff)
        const Mat3 R_c2v_eff = mat3_mul_mat3(R_unlevel, R_c2v_pitch);
        const Vec3 T_c2v_eff = mat3_mul_vec3(R_unlevel, cfg_.T_c2v);
        R_v2c_cur_ = mat3_transpose(R_c2v_eff);
        const Vec3 t = mat3_mul_vec3(R_v2c_cur_, T_c2v_eff);
        T_v2c_cur_ = {-t[0], -t[1], -t[2]};
    }

    // 车辆系地面点 (X, 0, Z) → 像素 (col, row); 相机后方返回 (-1, -1)
    std::pair<int, int> vehicle_xz_to_pixel(double x_m, double z_m) const {
        const Vec3 rot = mat3_mul_vec3(R_v2c_cur_, {x_m, 0.0, z_m});
        const Vec3 pt_c{rot[0] + T_v2c_cur_[0],
                        rot[1] + T_v2c_cur_[1],
                        rot[2] + T_v2c_cur_[2]};
        const double pz = pt_c[2];
        if (pz <= 0.0) return {-1, -1};
        const double uv0 = cfg_.K_final[0] * pt_c[0] + cfg_.K_final[1] * pt_c[1]
                           + cfg_.K_final[2] * pz;
        const double uv1 = cfg_.K_final[3] * pt_c[0] + cfg_.K_final[4] * pt_c[1]
                           + cfg_.K_final[5] * pz;
        return {static_cast<int>(round_half_even(uv0 / pz)),
                static_cast<int>(round_half_even(uv1 / pz))};
    }

    // 像素 (col, row) → 车辆系地面点 (X, Z); 与地平面 Y=0 求交。
    // 点在相机后方或射线与地面平行时返回无效。
    std::optional<std::pair<double, double>> pixel_to_vehicle_xz(
            double col, double row) const {
        // d = K⁻¹·(col, row, 1): K 上三角 → 回代求解
        const double d2 = 1.0;
        const double d1 = (row - cfg_.K_final[5] * d2) / cfg_.K_final[4];
        const double d0 = (col - cfg_.K_final[2] * d2) / cfg_.K_final[0];
        // (K[0][1] 畸变后恒为 0, 上式按无倾斜内参回代)
        const Vec3 d{d0, d1, d2};

        // R_T = R_v2cᵀ; r_y = R_T 第 1 行 (车辆系 Y 轴在相机系中的方向)
        const Mat3 R_T = mat3_transpose(R_v2c_cur_);
        const Vec3 r_y{R_T[3], R_T[4], R_T[5]};

        double denom = 0.0, rT = 0.0;
        for (int i = 0; i < 3; ++i) {
            denom += r_y[i] * d[i];
            rT += r_y[i] * T_v2c_cur_[i];
        }
        if (std::fabs(denom) < 1e-9) return std::nullopt;
        const double t = rT / denom;
        if (t <= 0.0) return std::nullopt;
        Vec3 pt_v{};
        for (int i = 0; i < 3; ++i)
            pt_v[i] = R_T[i * 3 + 0] * (t * d[0] - T_v2c_cur_[0])
                    + R_T[i * 3 + 1] * (t * d[1] - T_v2c_cur_[1])
                    + R_T[i * 3 + 2] * (t * d[2] - T_v2c_cur_[2]);
        return std::make_pair(pt_v[0], pt_v[2]);
    }

private:
    // nominal 外参: R_c2v 由 camera_rad 构造 (与 CameraInfo 同式)
    void reset_extrinsics() {
        const double c = std::cos(cfg_.camera_rad);
        const double s = std::sin(cfg_.camera_rad);
        const Mat3 R_c2v = {1.0, 0.0, 0.0,
                            0.0,  c,   s,
                            0.0, -s,   c};
        R_v2c_cur_ = mat3_transpose(R_c2v);
        const Vec3 t = mat3_mul_vec3(R_v2c_cur_, cfg_.T_c2v);
        T_v2c_cur_ = {-t[0], -t[1], -t[2]};
        roll_ = 0.0;
        pitch_ = 0.0;
    }

    Config cfg_;
    Mat3 R_v2c_cur_{};   // 当前生效外参 (set_attitude 后更新)
    Vec3 T_v2c_cur_{};
    double roll_ = 0.0;
    double pitch_ = 0.0;
};

}  // namespace rowmap
