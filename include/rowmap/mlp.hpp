// mlp.hpp — 三层 MLP 部署前向 (C3 质量网 / C5 行尾网共用)
//
// 与训练侧 numpy runtime (nav_simulator/dl/c3_runtime.py, c5_runtime.py)
// 逐算式对齐: 输入标准化 → Linear(32)+SiLU → Linear(16)+SiLU →
// Linear(out)。计算全程 float32 (镜像 numpy float32 matmul; 累加序
// 差异 ~1e-7 相对量级, 由逐项对照容差表达), 输出转 double (参考实现侧世界
// 坐标是 double, 不让网络内部精度决定输出 dtype)。
//
// 权重文件 .wts (小端二进制, tools/export_weights.py 从训练落盘
// npz 生成):
//   magic "RMW1" | u32 version=1 | u32 n_tensors
//   每张量: u32 name_len | name | u32 ndim | u32 dims[ndim] | f32 data
// 张量名: W1 (h1,in) / b1 (h1) / W2 (h2,h1) / b2 (h2) /
//         W3 (out,h2) / b3 (out) / feat_mean (in) / feat_std (in)

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace rowmap {

struct MlpNet {
    int in_dim = 0;
    int h1 = 0;
    int h2 = 0;
    int out_dim = 0;
    // 行主序权重 (与 npz 同布局)
    std::vector<float> w1, b1, w2, b2, w3, b3;
    std::vector<float> feat_mean, feat_std;

    bool valid() const { return in_dim > 0; }

    // 加载失败保持无效态 (调用方以 valid() 判断)。h1/h2/out 由
    // 权重文件自述, in/out 维的语义校验 (C3: 14→2, C5: 22→1)
    // 由 nav_core 的 setter 做
    //
    // 路径版供原生/板端调用 (Linux UTF-8 locale); Windows 中文系统
    // ANSI 代码页 (GBK) 下 pybind 传入的 UTF-8 字节路径打不开非
    // ASCII 文件名 —— pybind 绑定层用 Python I/O 读文件转走
    // load_buffer (跨平台零分支)
    bool load(const std::string& path) {
        std::FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) return false;
        std::vector<unsigned char> buf;
        unsigned char tmp[4096];
        std::size_t n = 0;
        while ((n = std::fread(tmp, 1, sizeof tmp, fp)) > 0)
            buf.insert(buf.end(), tmp, tmp + n);
        const bool ok = (std::ferror(fp) == 0)
                        && load_buffer(buf.data(), buf.size());
        std::fclose(fp);
        if (!ok) *this = MlpNet{};
        return ok;
    }

    bool load_buffer(const unsigned char* d, std::size_t n) {
        if (d == nullptr || n == 0) return false;
        std::size_t off = 0;
        auto take = [&](void* dst, std::size_t len) -> bool {
            if (off + len > n) return false;
            std::memcpy(dst, d + off, len);
            off += len;
            return true;
        };
        char magic[4];
        if (!take(magic, 4) || std::memcmp(magic, "RMW1", 4) != 0)
            return false;
        std::uint32_t version = 0, n_tensors = 0;
        if (!take(&version, 4) || version != 1) return false;
        if (!take(&n_tensors, 4) || n_tensors == 0 || n_tensors > 64)
            return false;
        std::unordered_map<std::string, Tensor> tensors;
        for (std::uint32_t t = 0; t < n_tensors; ++t) {
            std::uint32_t name_len = 0;
            if (!take(&name_len, 4) || name_len == 0 || name_len > 32)
                return false;
            std::string name(name_len, '\0');
            if (!take(&name[0], name_len)) return false;
            std::uint32_t ndim = 0;
            if (!take(&ndim, 4) || ndim == 0 || ndim > 2) return false;
            Tensor ten;
            ten.dims.resize(ndim);
            std::size_t count = 1;
            for (std::uint32_t dm = 0; dm < ndim; ++dm) {
                if (!take(&ten.dims[dm], 4)) return false;
                count *= ten.dims[dm];
            }
            if (count == 0 || count > (1u << 24)) return false;
            ten.data.resize(count);
            if (!take(ten.data.data(), count * sizeof(float)))
                return false;
            tensors[name] = std::move(ten);
        }
        if (off != n) return false;   // 尾部多余字节 = 格式错
        return assign(tensors);
    }

    // feats: in_dim 个 double → out: out_dim 个 double (线性输出,
    // 输出域激活由调用方解释: C3 out[1]=log σ² → exp+限幅,
    // C5 out[0]=logit → sigmoid)
    void forward(const double* feats, double* out) const {
        std::vector<float> x(static_cast<std::size_t>(in_dim));
        std::vector<float> h(static_cast<std::size_t>(h1));
        std::vector<float> g(static_cast<std::size_t>(h2));
        for (int i = 0; i < in_dim; ++i) {
            const float fx = static_cast<float>(feats[i]);
            x[static_cast<std::size_t>(i)] =
                (fx - feat_mean[static_cast<std::size_t>(i)])
                / feat_std[static_cast<std::size_t>(i)];
        }
        for (int j = 0; j < h1; ++j) {
            const float* wr = &w1[static_cast<std::size_t>(j) * in_dim];
            float acc = b1[static_cast<std::size_t>(j)];
            for (int i = 0; i < in_dim; ++i)
                acc += wr[i] * x[static_cast<std::size_t>(i)];
            h[static_cast<std::size_t>(j)] = acc / (1.0f + std::exp(-acc));
        }
        for (int j = 0; j < h2; ++j) {
            const float* wr = &w2[static_cast<std::size_t>(j) * h1];
            float acc = b2[static_cast<std::size_t>(j)];
            for (int i = 0; i < h1; ++i)
                acc += wr[i] * h[static_cast<std::size_t>(i)];
            g[static_cast<std::size_t>(j)] = acc / (1.0f + std::exp(-acc));
        }
        for (int k = 0; k < out_dim; ++k) {
            const float* wr = &w3[static_cast<std::size_t>(k) * h2];
            float acc = b3[static_cast<std::size_t>(k)];
            for (int j = 0; j < h2; ++j)
                acc += wr[j] * g[static_cast<std::size_t>(j)];
            out[k] = static_cast<double>(acc);
        }
    }

private:
    struct Tensor {
        std::vector<std::uint32_t> dims;
        std::vector<float> data;
    };

    static bool mat2(const std::unordered_map<std::string, Tensor>& ts,
                     const char* name, std::uint32_t rows,
                     std::uint32_t cols, std::vector<float>* out) {
        const auto it = ts.find(name);
        if (it == ts.end() || it->second.dims.size() != 2
                || it->second.dims[0] != rows
                || it->second.dims[1] != cols)
            return false;
        *out = it->second.data;
        return true;
    }
    static bool vec1(const std::unordered_map<std::string, Tensor>& ts,
                     const char* name, std::uint32_t n,
                     std::vector<float>* out) {
        const auto it = ts.find(name);
        if (it == ts.end() || it->second.dims.size() != 1
                || it->second.dims[0] != n)
            return false;
        *out = it->second.data;
        return true;
    }

    bool assign(
            const std::unordered_map<std::string, Tensor>& ts) {
        // 维度链自洽: in → h1 → h2 → out 从权重形状反推
        const auto it_w1 = ts.find("W1");
        const auto it_w3 = ts.find("W3");
        if (it_w1 == ts.end() || it_w3 == ts.end()
                || it_w1->second.dims.size() != 2
                || it_w3->second.dims.size() != 2)
            return false;
        const int in = static_cast<int>(it_w1->second.dims[1]);
        const int h1 = static_cast<int>(it_w1->second.dims[0]);
        const int out = static_cast<int>(it_w3->second.dims[0]);
        const int h2 = static_cast<int>(it_w3->second.dims[1]);
        if (in <= 0 || h1 <= 0 || h2 <= 0 || out <= 0) return false;
        if (!mat2(ts, "W1", h1, in, &w1)) return false;
        if (!vec1(ts, "b1", h1, &b1)) return false;
        if (!mat2(ts, "W2", h2, h1, &w2)) return false;
        if (!vec1(ts, "b2", h2, &b2)) return false;
        if (!mat2(ts, "W3", out, h2, &w3)) return false;
        if (!vec1(ts, "b3", out, &b3)) return false;
        if (!vec1(ts, "feat_mean", in, &feat_mean)) return false;
        if (!vec1(ts, "feat_std", in, &feat_std)) return false;
        for (float v : feat_std)
            if (!(v > 0.0f)) return false;
        in_dim = in; this->h1 = h1; this->h2 = h2; out_dim = out;
        return true;
    }
};

}  // namespace rowmap
