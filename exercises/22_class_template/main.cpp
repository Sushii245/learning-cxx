#include "../exercise.h"
#include <cstring>
// READ: 类模板 <https://zh.cppreference.com/w/cpp/language/class_template>

template<class T>
struct Tensor4D {
    unsigned int shape[4];
    T *data;

    Tensor4D(unsigned int const shape_[4], T const *data_) {
        unsigned int size = 1;
        // TODO: 填入正确的 shape 并计算 size
        for (int i = 0; i < 4; ++i) {
                    shape[i] = shape_[i];
                    size *= shape[i];
                }
        data = new T[size];
        std::memcpy(data, data_, size * sizeof(T));
    }
    ~Tensor4D() {
        delete[] data;
    }

    // 为了保持简单，禁止复制和移动
    Tensor4D(Tensor4D const &) = delete;
    Tensor4D(Tensor4D &&) noexcept = delete;

    // 这个加法需要支持“单向广播”。
    // 具体来说，`others` 可以具有与 `this` 不同的形状，形状不同的维度长度必须为 1。
    // `others` 长度为 1 但 `this` 长度不为 1 的维度将发生广播计算。
    // 例如，`this` 形状为 `[1, 2, 3, 4]`，`others` 形状为 `[1, 2, 1, 4]`，
    // 则 `this` 与 `others` 相加时，3 个形状为 `[1, 2, 1, 4]` 的子张量各自与 `others` 对应项相加。
    Tensor4D &operator+=(Tensor4D const &others) {
            for (int k = 0; k < 4; ++k) {
                ASSERT(others.shape[k] == 1 || others.shape[k] == shape[k], "broadcast shape mismatch");
            }

            unsigned int s0 = shape[0], s1 = shape[1], s2 = shape[2], s3 = shape[3];

            unsigned int o0 = others.shape[0], o1 = others.shape[1], o2 = others.shape[2], o3 = others.shape[3];

            unsigned int this_str0 = s1 * s2 * s3;
            unsigned int this_str1 = s2 * s3;
            unsigned int this_str2 = s3;
            unsigned int this_str3 = 1;

            unsigned int other_str0 = o1 * o2 * o3;
            unsigned int other_str1 = o2 * o3;
            unsigned int other_str2 = o3;
            unsigned int other_str3 = 1;

            unsigned int other_step0 = (o0 == 1) ? 0 : other_str0;
            unsigned int other_step1 = (o1 == 1) ? 0 : other_str1;
            unsigned int other_step2 = (o2 == 1) ? 0 : other_str2;
            unsigned int other_step3 = (o3 == 1) ? 0 : other_str3;

            for (unsigned int i0 = 0; i0 < s0; ++i0) {
                for (unsigned int i1 = 0; i1 < s1; ++i1) {
                    for (unsigned int i2 = 0; i2 < s2; ++i2) {
                        for (unsigned int i3 = 0; i3 < s3; ++i3) {
                            unsigned int idx_this =
                                i0 * this_str0 + i1 * this_str1 + i2 * this_str2 + i3 * this_str3;
                            unsigned int idx_other =
                                i0 * other_step0 + i1 * other_step1 + i2 * other_step2 + i3 * other_step3;
                            data[idx_this] += others.data[idx_other];
                        }
                    }
                }
            }

            return *this;
        }
};

// ---- 不要修改以下代码 ----
int main(int argc, char **argv) {
    {
        unsigned int shape[]{1, 2, 3, 4};
        // clang-format off
        int data[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        auto t0 = Tensor4D<int>(shape, data);
        auto t1 = Tensor4D<int>(shape, data);

        t0 += t1;
        for (auto i = 0u; i < sizeof(data) / sizeof(*data); ++i) {
            ASSERT(t0.data[i] == data[i] * 2, "Tensor doubled by plus its self.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        float d0[]{
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,

            4, 4, 4, 4,
            5, 5, 5, 5,
            6, 6, 6, 6};
        // clang-format on
        unsigned int s1[]{1, 2, 3, 1};
        // clang-format off
        float d1[]{
            6,
            5,
            4,

            3,
            2,
            1};
        // clang-format on

        auto t0 = Tensor4D<float>(s0, d0);
        auto t1 = Tensor4D<float>(s1, d1);

        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == 7.f, "Every element of t0 should be 7 after adding t1 to it.");
        }
    }
    {
        unsigned int s0[]{1, 2, 3, 4};
        // clang-format off
        double d0[]{
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,

            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24};
        // clang-format on
        unsigned int s1[]{1, 1, 1, 1};
        double d1[]{1};

        auto t0 = Tensor4D<double>(s0, d0);
        auto t1 = Tensor4D<double>(s1, d1);

        t0 += t1;
        for (auto i = 0u; i < sizeof(d0) / sizeof(*d0); ++i) {
            ASSERT(t0.data[i] == d0[i] + 1, "Every element of t0 should be incremented by 1 after adding t1 to it.");
        }
    }
}
