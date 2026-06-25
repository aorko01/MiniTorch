#include "autograd/ops/Ops.h"
#include "Tensor.h"
#include <iostream>
#include <chrono>
#include <cassert>
#include <cmath>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// ── helpers ──────────────────────────────────────────────────────────────────

static bool approx_equal(float a, float b, float eps = 1e-5f)
{
    return std::fabs(a - b) < eps;
}

// Time a callable and return elapsed milliseconds
template <typename Fn>
static double time_ms(Fn &&fn)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static int passed = 0;
static int failed = 0;

#define ASSERT_TRUE(cond, msg)                     \
    do                                             \
    {                                              \
        if (!(cond))                               \
        {                                          \
            std::cerr << "[FAIL] " << msg << "\n"; \
            ++failed;                              \
        }                                          \
        else                                       \
        {                                          \
            std::cout << "[PASS] " << msg << "\n"; \
            ++passed;                              \
        }                                          \
    } while (0)

// ── add tests ────────────────────────────────────────────────────────────────

void test_add_basic()
{
    Tensor a({3}), b({3});
    a.set({0}, 1.0f);
    a.set({1}, 2.0f);
    a.set({2}, 3.0f);
    b.set({0}, 4.0f);
    b.set({1}, 5.0f);
    b.set({2}, 6.0f);

    Tensor c = add(a, b);

    ASSERT_TRUE(approx_equal(c.get({0}), 5.0f), "add basic [0] == 5");
    ASSERT_TRUE(approx_equal(c.get({1}), 7.0f), "add basic [1] == 7");
    ASSERT_TRUE(approx_equal(c.get({2}), 9.0f), "add basic [2] == 9");
}

void test_add_2d()
{
    Tensor a({2, 2}), b({2, 2});
    a.set({0, 0}, 1.f);
    a.set({0, 1}, 2.f);
    a.set({1, 0}, 3.f);
    a.set({1, 1}, 4.f);

    b.set({0, 0}, 10.f);
    b.set({0, 1}, 20.f);
    b.set({1, 0}, 30.f);
    b.set({1, 1}, 40.f);

    Tensor c = add(a, b);

    ASSERT_TRUE(approx_equal(c.get({0, 0}), 11.f), "add 2d [0][0] == 11");
    ASSERT_TRUE(approx_equal(c.get({0, 1}), 22.f), "add 2d [0][1] == 22");
    ASSERT_TRUE(approx_equal(c.get({1, 0}), 33.f), "add 2d [1][0] == 33");
    ASSERT_TRUE(approx_equal(c.get({1, 1}), 44.f), "add 2d [1][1] == 44");
}

void test_add_zeros()
{
    Tensor a({4}), b({4});
    for (int i = 0; i < 4; ++i)
    {
        a.set({i}, static_cast<float>(i));
        b.set({i}, 0.0f);
    }
    Tensor c = add(a, b);
    for (int i = 0; i < 4; ++i)
        ASSERT_TRUE(approx_equal(c.get({i}), static_cast<float>(i)),
                    "add zero identity [" + std::to_string(i) + "]");
}

void test_add_shape_mismatch()
{
    Tensor a({3}), b({4});
    bool threw = false;
    try
    {
        add(a, b);
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    ASSERT_TRUE(threw, "add shape mismatch throws");
}

// ── mul tests ────────────────────────────────────────────────────────────────

void test_mul_basic()
{
    Tensor a({3}), b({3});
    a.set({0}, 2.0f);
    a.set({1}, 3.0f);
    a.set({2}, 4.0f);
    b.set({0}, 5.0f);
    b.set({1}, 6.0f);
    b.set({2}, 7.0f);

    Tensor c = mul(a, b);

    ASSERT_TRUE(approx_equal(c.get({0}), 10.f), "mul basic [0] == 10");
    ASSERT_TRUE(approx_equal(c.get({1}), 18.f), "mul basic [1] == 18");
    ASSERT_TRUE(approx_equal(c.get({2}), 28.f), "mul basic [2] == 28");
}

void test_mul_2d()
{
    Tensor a({2, 2}), b({2, 2});
    a.set({0, 0}, 1.f);
    a.set({0, 1}, 2.f);
    a.set({1, 0}, 3.f);
    a.set({1, 1}, 4.f);

    b.set({0, 0}, 2.f);
    b.set({0, 1}, 3.f);
    b.set({1, 0}, 4.f);
    b.set({1, 1}, 5.f);

    Tensor c = mul(a, b);

    ASSERT_TRUE(approx_equal(c.get({0, 0}), 2.f), "mul 2d [0][0] == 2");
    ASSERT_TRUE(approx_equal(c.get({0, 1}), 6.f), "mul 2d [0][1] == 6");
    ASSERT_TRUE(approx_equal(c.get({1, 0}), 12.f), "mul 2d [1][0] == 12");
    ASSERT_TRUE(approx_equal(c.get({1, 1}), 20.f), "mul 2d [1][1] == 20");
}

void test_mul_by_one()
{
    Tensor a({4}), b({4});
    for (int i = 0; i < 4; ++i)
    {
        a.set({i}, static_cast<float>(i + 1));
        b.set({i}, 1.0f);
    }
    Tensor c = mul(a, b);
    for (int i = 0; i < 4; ++i)
        ASSERT_TRUE(approx_equal(c.get({i}), static_cast<float>(i + 1)),
                    "mul by one identity [" + std::to_string(i) + "]");
}

void test_mul_by_zero()
{
    Tensor a({4}), b({4});
    for (int i = 0; i < 4; ++i)
    {
        a.set({i}, static_cast<float>(i + 1));
        b.set({i}, 0.0f);
    }
    Tensor c = mul(a, b);
    for (int i = 0; i < 4; ++i)
        ASSERT_TRUE(approx_equal(c.get({i}), 0.f),
                    "mul by zero [" + std::to_string(i) + "] == 0");
}

void test_mul_shape_mismatch()
{
    Tensor a({3}), b({5});
    bool threw = false;
    try
    {
        mul(a, b);
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    ASSERT_TRUE(threw, "mul shape mismatch throws");
}

#ifdef USE_CUDA
void test_add_mul_cuda_contiguous()
{
    Tensor a({4}), b({4});
    for (int i = 0; i < 4; ++i)
    {
        a.set({i}, static_cast<float>(i + 1));
        b.set({i}, static_cast<float>((i + 1) * 2));
    }

    a.set_device("cuda");
    b.set_device("cuda");

    Tensor sum = add(a, b);
    Tensor product = mul(a, b);

    for (int i = 0; i < 4; ++i)
    {
        ASSERT_TRUE(approx_equal(sum.get({i}), static_cast<float>((i + 1) * 3)),
                    "cuda add contiguous [" + std::to_string(i) + "]");
        ASSERT_TRUE(approx_equal(product.get({i}), static_cast<float>((i + 1) * ((i + 1) * 2))),
                    "cuda mul contiguous [" + std::to_string(i) + "]");
    }
}
#endif

// ── large tensor correctness test ────────────────────────────────────────────

void test_add_mul_large_tensor()
{
    const int N = 10000;

    Tensor a({N}), b({N});

    for (int i = 0; i < N; ++i)
    {
        a.set({i}, static_cast<float>(i));
        b.set({i}, static_cast<float>(i % 7));
    }

    // add
    Tensor c = add(a, b);

    for (int i = 0; i < N; ++i)
    {
        float expected = static_cast<float>(i + (i % 7));
        ASSERT_TRUE(approx_equal(c.get({i}), expected),
                    "large add [" + std::to_string(i) + "]");
    }

    // mul
    Tensor d = mul(a, b);

    for (int i = 0; i < N; ++i)
    {
        float expected = static_cast<float>(i * (i % 7));
        ASSERT_TRUE(approx_equal(d.get({i}), expected),
                    "large mul [" + std::to_string(i) + "]");
    }
}

// ── broadcast tests ───────────────────────────────────────────────────────────
//
// Naming convention: test_{op}_broadcast_{pattern}
//   scalar      – {1}    op {N}    → {N}
//   row         – {1,N}  op {M,N}  → {M,N}
//   col         – {M,1}  op {M,N}  → {M,N}
//   outer       – {M,1}  op {1,N}  → {M,N}
//   leading_dim – {N}    op {M,N}  → {M,N}   (NumPy-style trailing alignment)

// ── scalar broadcast  {1} op {N} ─────────────────────────────────────────────

void test_add_broadcast_scalar()
{
    Tensor a({1}), b({4});
    a.set({0}, 10.f);
    for (int i = 0; i < 4; ++i)
        b.set({i}, static_cast<float>(i + 1));

    Tensor c = add(a, b);

    for (int i = 0; i < 4; ++i)
        ASSERT_TRUE(approx_equal(c.get({i}), 10.f + static_cast<float>(i + 1)),
                    "broadcast scalar add [" + std::to_string(i) + "]");
}

void test_mul_broadcast_scalar()
{
    Tensor a({1}), b({4});
    a.set({0}, 3.f);
    for (int i = 0; i < 4; ++i)
        b.set({i}, static_cast<float>(i + 1));

    Tensor c = mul(a, b);

    for (int i = 0; i < 4; ++i)
        ASSERT_TRUE(approx_equal(c.get({i}), 3.f * static_cast<float>(i + 1)),
                    "broadcast scalar mul [" + std::to_string(i) + "]");
}

void test_add_broadcast_scalar_reversed()
{
    Tensor a({4}), b({1});
    for (int i = 0; i < 4; ++i)
        a.set({i}, static_cast<float>(i + 1));
    b.set({0}, 10.f);

    Tensor c = add(a, b);

    for (int i = 0; i < 4; ++i)
        ASSERT_TRUE(approx_equal(c.get({i}), static_cast<float>(i + 1) + 10.f),
                    "broadcast scalar add reversed [" + std::to_string(i) + "]");
}

void test_mul_broadcast_scalar_reversed()
{
    Tensor a({4}), b({1});
    for (int i = 0; i < 4; ++i)
        a.set({i}, static_cast<float>(i + 1));
    b.set({0}, 3.f);

    Tensor c = mul(a, b);

    for (int i = 0; i < 4; ++i)
        ASSERT_TRUE(approx_equal(c.get({i}), static_cast<float>(i + 1) * 3.f),
                    "broadcast scalar mul reversed [" + std::to_string(i) + "]");
}

// ── row broadcast  {1,N} op {M,N} ────────────────────────────────────────────

void test_add_broadcast_row()
{
    Tensor a({1, 3}), b({2, 3});
    a.set({0, 0}, 1.f);
    a.set({0, 1}, 2.f);
    a.set({0, 2}, 3.f);

    b.set({0, 0}, 10.f);
    b.set({0, 1}, 20.f);
    b.set({0, 2}, 30.f);
    b.set({1, 0}, 40.f);
    b.set({1, 1}, 50.f);
    b.set({1, 2}, 60.f);

    Tensor c = add(a, b);

    ASSERT_TRUE(approx_equal(c.get({0, 0}), 11.f), "broadcast row add [0][0]");
    ASSERT_TRUE(approx_equal(c.get({0, 1}), 22.f), "broadcast row add [0][1]");
    ASSERT_TRUE(approx_equal(c.get({0, 2}), 33.f), "broadcast row add [0][2]");
    ASSERT_TRUE(approx_equal(c.get({1, 0}), 41.f), "broadcast row add [1][0]");
    ASSERT_TRUE(approx_equal(c.get({1, 1}), 52.f), "broadcast row add [1][1]");
    ASSERT_TRUE(approx_equal(c.get({1, 2}), 63.f), "broadcast row add [1][2]");
}

void test_mul_broadcast_row()
{
    Tensor a({1, 3}), b({2, 3});
    a.set({0, 0}, 2.f);
    a.set({0, 1}, 3.f);
    a.set({0, 2}, 4.f);

    b.set({0, 0}, 1.f);
    b.set({0, 1}, 2.f);
    b.set({0, 2}, 3.f);
    b.set({1, 0}, 4.f);
    b.set({1, 1}, 5.f);
    b.set({1, 2}, 6.f);

    Tensor c = mul(a, b);

    ASSERT_TRUE(approx_equal(c.get({0, 0}), 2.f), "broadcast row mul [0][0]");
    ASSERT_TRUE(approx_equal(c.get({0, 1}), 6.f), "broadcast row mul [0][1]");
    ASSERT_TRUE(approx_equal(c.get({0, 2}), 12.f), "broadcast row mul [0][2]");
    ASSERT_TRUE(approx_equal(c.get({1, 0}), 8.f), "broadcast row mul [1][0]");
    ASSERT_TRUE(approx_equal(c.get({1, 1}), 15.f), "broadcast row mul [1][1]");
    ASSERT_TRUE(approx_equal(c.get({1, 2}), 24.f), "broadcast row mul [1][2]");
}

// ── column broadcast  {M,1} op {M,N} ─────────────────────────────────────────

void test_add_broadcast_col()
{
    Tensor a({3, 1}), b({3, 2});
    a.set({0, 0}, 1.f);
    a.set({1, 0}, 2.f);
    a.set({2, 0}, 3.f);

    b.set({0, 0}, 10.f);
    b.set({0, 1}, 20.f);
    b.set({1, 0}, 30.f);
    b.set({1, 1}, 40.f);
    b.set({2, 0}, 50.f);
    b.set({2, 1}, 60.f);

    Tensor c = add(a, b);

    ASSERT_TRUE(approx_equal(c.get({0, 0}), 11.f), "broadcast col add [0][0]");
    ASSERT_TRUE(approx_equal(c.get({0, 1}), 21.f), "broadcast col add [0][1]");
    ASSERT_TRUE(approx_equal(c.get({1, 0}), 32.f), "broadcast col add [1][0]");
    ASSERT_TRUE(approx_equal(c.get({1, 1}), 42.f), "broadcast col add [1][1]");
    ASSERT_TRUE(approx_equal(c.get({2, 0}), 53.f), "broadcast col add [2][0]");
    ASSERT_TRUE(approx_equal(c.get({2, 1}), 63.f), "broadcast col add [2][1]");
}

void test_mul_broadcast_col()
{
    Tensor a({3, 1}), b({3, 2});
    a.set({0, 0}, 2.f);
    a.set({1, 0}, 3.f);
    a.set({2, 0}, 4.f);

    b.set({0, 0}, 5.f);
    b.set({0, 1}, 6.f);
    b.set({1, 0}, 7.f);
    b.set({1, 1}, 8.f);
    b.set({2, 0}, 9.f);
    b.set({2, 1}, 10.f);

    Tensor c = mul(a, b);

    ASSERT_TRUE(approx_equal(c.get({0, 0}), 10.f), "broadcast col mul [0][0]");
    ASSERT_TRUE(approx_equal(c.get({0, 1}), 12.f), "broadcast col mul [0][1]");
    ASSERT_TRUE(approx_equal(c.get({1, 0}), 21.f), "broadcast col mul [1][0]");
    ASSERT_TRUE(approx_equal(c.get({1, 1}), 24.f), "broadcast col mul [1][1]");
    ASSERT_TRUE(approx_equal(c.get({2, 0}), 36.f), "broadcast col mul [2][0]");
    ASSERT_TRUE(approx_equal(c.get({2, 1}), 40.f), "broadcast col mul [2][1]");
}

// ── outer-product expand  {M,1} op {1,N} → {M,N} ────────────────────────────

void test_add_broadcast_outer()
{
    Tensor a({3, 1}), b({1, 4});
    a.set({0, 0}, 1.f);
    a.set({1, 0}, 2.f);
    a.set({2, 0}, 3.f);
    b.set({0, 0}, 10.f);
    b.set({0, 1}, 20.f);
    b.set({0, 2}, 30.f);
    b.set({0, 3}, 40.f);

    Tensor c = add(a, b);

    ASSERT_TRUE(approx_equal(c.get({0, 0}), 11.f), "broadcast outer add [0][0]");
    ASSERT_TRUE(approx_equal(c.get({0, 3}), 41.f), "broadcast outer add [0][3]");
    ASSERT_TRUE(approx_equal(c.get({1, 1}), 22.f), "broadcast outer add [1][1]");
    ASSERT_TRUE(approx_equal(c.get({2, 2}), 33.f), "broadcast outer add [2][2]");
    ASSERT_TRUE(approx_equal(c.get({2, 3}), 43.f), "broadcast outer add [2][3]");
}

void test_mul_broadcast_outer()
{
    Tensor a({3, 1}), b({1, 4});
    a.set({0, 0}, 2.f);
    a.set({1, 0}, 3.f);
    a.set({2, 0}, 4.f);
    b.set({0, 0}, 1.f);
    b.set({0, 1}, 2.f);
    b.set({0, 2}, 3.f);
    b.set({0, 3}, 4.f);

    Tensor c = mul(a, b);

    ASSERT_TRUE(approx_equal(c.get({0, 0}), 2.f), "broadcast outer mul [0][0]");
    ASSERT_TRUE(approx_equal(c.get({0, 3}), 8.f), "broadcast outer mul [0][3]");
    ASSERT_TRUE(approx_equal(c.get({1, 1}), 6.f), "broadcast outer mul [1][1]");
    ASSERT_TRUE(approx_equal(c.get({2, 2}), 12.f), "broadcast outer mul [2][2]");
    ASSERT_TRUE(approx_equal(c.get({2, 3}), 16.f), "broadcast outer mul [2][3]");
}

// ── leading-dim expand  {N} op {M,N} → {M,N} ─────────────────────────────────

void test_add_broadcast_leading_dim()
{
    Tensor a({3}), b({2, 3});
    a.set({0}, 1.f);
    a.set({1}, 2.f);
    a.set({2}, 3.f);

    b.set({0, 0}, 10.f);
    b.set({0, 1}, 20.f);
    b.set({0, 2}, 30.f);
    b.set({1, 0}, 40.f);
    b.set({1, 1}, 50.f);
    b.set({1, 2}, 60.f);

    Tensor c = add(a, b);

    ASSERT_TRUE(approx_equal(c.get({0, 0}), 11.f), "broadcast leading add [0][0]");
    ASSERT_TRUE(approx_equal(c.get({0, 1}), 22.f), "broadcast leading add [0][1]");
    ASSERT_TRUE(approx_equal(c.get({0, 2}), 33.f), "broadcast leading add [0][2]");
    ASSERT_TRUE(approx_equal(c.get({1, 0}), 41.f), "broadcast leading add [1][0]");
    ASSERT_TRUE(approx_equal(c.get({1, 1}), 52.f), "broadcast leading add [1][1]");
    ASSERT_TRUE(approx_equal(c.get({1, 2}), 63.f), "broadcast leading add [1][2]");
}

void test_mul_broadcast_leading_dim()
{
    Tensor a({3}), b({2, 3});
    a.set({0}, 2.f);
    a.set({1}, 3.f);
    a.set({2}, 4.f);

    b.set({0, 0}, 1.f);
    b.set({0, 1}, 2.f);
    b.set({0, 2}, 3.f);
    b.set({1, 0}, 4.f);
    b.set({1, 1}, 5.f);
    b.set({1, 2}, 6.f);

    Tensor c = mul(a, b);

    ASSERT_TRUE(approx_equal(c.get({0, 0}), 2.f), "broadcast leading mul [0][0]");
    ASSERT_TRUE(approx_equal(c.get({0, 1}), 6.f), "broadcast leading mul [0][1]");
    ASSERT_TRUE(approx_equal(c.get({0, 2}), 12.f), "broadcast leading mul [0][2]");
    ASSERT_TRUE(approx_equal(c.get({1, 0}), 8.f), "broadcast leading mul [1][0]");
    ASSERT_TRUE(approx_equal(c.get({1, 1}), 15.f), "broadcast leading mul [1][1]");
    ASSERT_TRUE(approx_equal(c.get({1, 2}), 24.f), "broadcast leading mul [1][2]");
}

// ── broadcast error / rejection cases ────────────────────────────────────────

void test_add_broadcast_incompatible()
{
    Tensor a({2, 3}), b({2, 4});
    bool threw = false;
    try
    {
        add(a, b);
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    ASSERT_TRUE(threw, "broadcast add {2,3} vs {2,4} throws");
}

void test_mul_broadcast_incompatible()
{
    Tensor a({2, 3}), b({2, 4});
    bool threw = false;
    try
    {
        mul(a, b);
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    ASSERT_TRUE(threw, "broadcast mul {2,3} vs {2,4} throws");
}

void test_add_broadcast_incompatible_leading()
{
    Tensor a({3}), b({2, 4});
    bool threw = false;
    try
    {
        add(a, b);
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    ASSERT_TRUE(threw, "broadcast add {3} vs {2,4} throws");
}

void test_mul_broadcast_incompatible_leading()
{
    Tensor a({3}), b({2, 4});
    bool threw = false;
    try
    {
        mul(a, b);
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    ASSERT_TRUE(threw, "broadcast mul {3} vs {2,4} throws");
}

// ── broadcast identity / edge cases ──────────────────────────────────────────

void test_add_broadcast_with_zeros()
{
    Tensor a({1, 3}), b({4, 3});
    a.set({0, 0}, 0.f);
    a.set({0, 1}, 0.f);
    a.set({0, 2}, 0.f);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j)
            b.set({i, j}, static_cast<float>(i * 3 + j));

    Tensor c = add(a, b);

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j)
            ASSERT_TRUE(approx_equal(c.get({i, j}), static_cast<float>(i * 3 + j)),
                        "broadcast zero-row add identity [" + std::to_string(i) + "][" + std::to_string(j) + "]");
}

void test_mul_broadcast_with_ones()
{
    Tensor a({3, 1}), b({3, 4});
    a.set({0, 0}, 1.f);
    a.set({1, 0}, 1.f);
    a.set({2, 0}, 1.f);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            b.set({i, j}, static_cast<float>(i * 4 + j + 1));

    Tensor c = mul(a, b);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            ASSERT_TRUE(approx_equal(c.get({i, j}), static_cast<float>(i * 4 + j + 1)),
                        "broadcast ones-col mul identity [" + std::to_string(i) + "][" + std::to_string(j) + "]");
}

// ── timing benchmarks (CPU only) ─────────────────────────────────────────────

void bench_add(int size)
{
    std::vector<int> shape = {size};
    Tensor a(shape), b(shape);

    float *wa = a.get_tensor_unrolled();
    float *wb = b.get_tensor_unrolled();
    for (int i = 0; i < size; ++i)
    {
        wa[i] = static_cast<float>(i);
        wb[i] = 1.f;
    }

    double ms = time_ms([&]
                        { add(a, b); });
    std::cout << "[BENCH] add  size=" << size << "  time=" << ms << " ms\n";
}

void bench_mul(int size)
{
    std::vector<int> shape = {size};
    Tensor a(shape), b(shape);

    float *wa = a.get_tensor_unrolled();
    float *wb = b.get_tensor_unrolled();
    for (int i = 0; i < size; ++i)
    {
        wa[i] = static_cast<float>(i);
        wb[i] = 2.f;
    }

    double ms = time_ms([&]
                        { mul(a, b); });
    std::cout << "[BENCH] mul  size=" << size << "  time=" << ms << " ms\n";
}

void bench_add_broadcast_row(int rows, int cols)
{
    Tensor a({1, cols}), b({rows, cols});

    float *wa = a.get_tensor_unrolled();
    float *wb = b.get_tensor_unrolled();
    for (int j = 0; j < cols; ++j)
        wa[j] = static_cast<float>(j);
    for (int i = 0; i < rows * cols; ++i)
        wb[i] = static_cast<float>(i);

    double ms = time_ms([&]
                        { add(a, b); });
    std::cout << "[BENCH] add  broadcast_row  rows=" << rows
              << "  cols=" << cols << "  time=" << ms << " ms\n";
}

void bench_mul_broadcast_col(int rows, int cols)
{
    Tensor a({rows, 1}), b({rows, cols});

    float *wa = a.get_tensor_unrolled();
    float *wb = b.get_tensor_unrolled();
    for (int i = 0; i < rows; ++i)
        wa[i] = static_cast<float>(i + 1);
    for (int i = 0; i < rows * cols; ++i)
        wb[i] = static_cast<float>(i);

    double ms = time_ms([&]
                        { mul(a, b); });
    std::cout << "[BENCH] mul  broadcast_col  rows=" << rows
              << "  cols=" << cols << "  time=" << ms << " ms\n";
}

// ── CUDA vs CPU timing benchmarks ────────────────────────────────────────────
//
// Each function:
//   1. Fills identical data on both CPU and CUDA tensors.
//   2. Runs a warmup pass on CUDA to avoid measuring lazy-init overhead.
//   3. Times both devices and spot-checks results for correctness.
//   4. Prints a side-by-side summary with a speedup ratio.

#ifdef USE_CUDA

// Fill a raw buffer with: buf[i] = i * scale + 1.0
static void fill_pattern(float *buf, int n, float scale = 1.f)
{
    for (int i = 0; i < n; ++i)
        buf[i] = static_cast<float>(i) * scale + 1.f;
}

// ── 1-D add: CPU vs CUDA  {N} ────────────────────────────────────────────────

void bench_add_cpu_vs_cuda_1d(int N)
{
    std::cout << "\n[CUDA BENCH] add 1-D  N=" << N << "\n";

    // CPU
    Tensor a_cpu({N}), b_cpu({N});
    fill_pattern(a_cpu.get_tensor_unrolled(), N, 1.f);
    fill_pattern(b_cpu.get_tensor_unrolled(), N, 2.f);

    Tensor c_cpu({N});
    double cpu_ms = time_ms([&]
                            { c_cpu = add(a_cpu, b_cpu); });

    // spot-check: a[i] = i*1+1,  b[i] = i*2+1  →  c[i] = 3i + 2
    for (int i : {0, N / 4, N / 2, N - 1})
    {
        float expected = static_cast<float>(i) * 3.f + 2.f;
        ASSERT_TRUE(approx_equal(c_cpu.get({i}), expected),
                    "cpu add 1-D spot-check [" + std::to_string(i) + "]");
    }

    // CUDA: warmup then timed run
    Tensor a_cuda({N}), b_cuda({N});
    fill_pattern(a_cuda.get_tensor_unrolled(), N, 1.f);
    fill_pattern(b_cuda.get_tensor_unrolled(), N, 2.f);
    a_cuda.set_device("cuda");
    b_cuda.set_device("cuda");

    add(a_cuda, b_cuda);     // warmup
    cudaDeviceSynchronize(); // flush before timing

    Tensor c_cuda({N});
    double cuda_ms = time_ms([&]
                             {
                                 c_cuda = add(a_cuda, b_cuda);
                                 cudaDeviceSynchronize(); // wait for kernel to finish
                             });

    for (int i : {0, N / 4, N / 2, N - 1})
    {
        float expected = static_cast<float>(i) * 3.f + 2.f;
        ASSERT_TRUE(approx_equal(c_cuda.get({i}), expected),
                    "cuda add 1-D spot-check [" + std::to_string(i) + "]");
    }

    std::cout << "[CUDA BENCH] add 1-D  N=" << N
              << "  cpu=" << cpu_ms << " ms"
              << "  cuda=" << cuda_ms << " ms"
              << "  speedup=" << (cpu_ms / cuda_ms) << "x\n";
}

// ── 1-D mul: CPU vs CUDA  {N} ────────────────────────────────────────────────

void bench_mul_cpu_vs_cuda_1d(int N)
{
    std::cout << "\n[CUDA BENCH] mul 1-D  N=" << N << "\n";

    // CPU
    Tensor a_cpu({N}), b_cpu({N});
    fill_pattern(a_cpu.get_tensor_unrolled(), N, 1.f);
    fill_pattern(b_cpu.get_tensor_unrolled(), N, 0.5f);

    Tensor c_cpu({N});
    double cpu_ms = time_ms([&]
                            { c_cpu = mul(a_cpu, b_cpu); });

    // spot-check: a[i] = i*1+1,  b[i] = i*0.5+1  →  c[i] = (i+1)*(0.5i+1)
    for (int i : {0, N / 4, N / 2, N - 1})
    {
        float expected = (static_cast<float>(i) + 1.f) *
                         (static_cast<float>(i) * 0.5f + 1.f);
        ASSERT_TRUE(approx_equal(c_cpu.get({i}), expected, 1e-3f),
                    "cpu mul 1-D spot-check [" + std::to_string(i) + "]");
    }

    // CUDA: warmup then timed run
    Tensor a_cuda({N}), b_cuda({N});
    fill_pattern(a_cuda.get_tensor_unrolled(), N, 1.f);
    fill_pattern(b_cuda.get_tensor_unrolled(), N, 0.5f);
    a_cuda.set_device("cuda");
    b_cuda.set_device("cuda");

    mul(a_cuda, b_cuda);     // warmup
    cudaDeviceSynchronize(); // flush before timing

    Tensor c_cuda({N});
    double cuda_ms = time_ms([&]
                             {
                                 c_cuda = mul(a_cuda, b_cuda);
                                 cudaDeviceSynchronize(); // wait for kernel to finish
                             });

    for (int i : {0, N / 4, N / 2, N - 1})
    {
        float expected = (static_cast<float>(i) + 1.f) *
                         (static_cast<float>(i) * 0.5f + 1.f);
        ASSERT_TRUE(approx_equal(c_cuda.get({i}), expected, 1e-3f),
                    "cuda mul 1-D spot-check [" + std::to_string(i) + "]");
    }

    std::cout << "[CUDA BENCH] mul 1-D  N=" << N
              << "  cpu=" << cpu_ms << " ms"
              << "  cuda=" << cuda_ms << " ms"
              << "  speedup=" << (cpu_ms / cuda_ms) << "x\n";
}

// ── 2-D broadcast-row add: CPU vs CUDA  {1,C} op {R,C} ──────────────────────

void bench_add_broadcast_row_cpu_vs_cuda(int rows, int cols)
{
    std::cout << "\n[CUDA BENCH] add broadcast-row  rows=" << rows
              << "  cols=" << cols << "\n";

    // a[0][j] = j*1+1,  b[i][j] = (i*cols+j)*0.5+1
    // expected c[i][j] = a[0][j] + b[i][j]
    auto expected_val = [&](int r, int c2)
    {
        return (static_cast<float>(c2) * 1.f + 1.f) +
               (static_cast<float>(r * cols + c2) * 0.5f + 1.f);
    };

    // CPU
    Tensor a_cpu({1, cols}), b_cpu({rows, cols});
    fill_pattern(a_cpu.get_tensor_unrolled(), cols, 1.f);
    fill_pattern(b_cpu.get_tensor_unrolled(), rows * cols, 0.5f);

    Tensor c_cpu({rows, cols});
    double cpu_ms = time_ms([&]
                            { c_cpu = add(a_cpu, b_cpu); });

    for (auto [r, c2] : std::vector<std::pair<int, int>>{{0, 0}, {0, cols - 1}, {rows / 2, cols / 2}, {rows - 1, 0}, {rows - 1, cols - 1}})
        ASSERT_TRUE(approx_equal(c_cpu.get({r, c2}), expected_val(r, c2), 1e-3f),
                    "cpu broadcast-row add [" + std::to_string(r) + "][" + std::to_string(c2) + "]");

    // CUDA
    Tensor a_cuda({1, cols}), b_cuda({rows, cols});
    fill_pattern(a_cuda.get_tensor_unrolled(), cols, 1.f);
    fill_pattern(b_cuda.get_tensor_unrolled(), rows * cols, 0.5f);
    a_cuda.set_device("cuda");
    b_cuda.set_device("cuda");

    add(a_cuda, b_cuda);     // warmup
    cudaDeviceSynchronize(); // flush before timing

    Tensor c_cuda({rows, cols});
    double cuda_ms = time_ms([&]
                             {
                                 c_cuda = add(a_cuda, b_cuda);
                                 cudaDeviceSynchronize(); // wait for kernel to finish
                             });

    for (auto [r, c2] : std::vector<std::pair<int, int>>{{0, 0}, {0, cols - 1}, {rows / 2, cols / 2}, {rows - 1, 0}, {rows - 1, cols - 1}})
        ASSERT_TRUE(approx_equal(c_cuda.get({r, c2}), expected_val(r, c2), 1e-3f),
                    "cuda broadcast-row add [" + std::to_string(r) + "][" + std::to_string(c2) + "]");

    std::cout << "[CUDA BENCH] add broadcast-row  rows=" << rows
              << "  cols=" << cols
              << "  cpu=" << cpu_ms << " ms"
              << "  cuda=" << cuda_ms << " ms"
              << "  speedup=" << (cpu_ms / cuda_ms) << "x\n";
}

// ── 2-D broadcast-col mul: CPU vs CUDA  {R,1} op {R,C} ──────────────────────

void bench_mul_broadcast_col_cpu_vs_cuda(int rows, int cols)
{
    std::cout << "\n[CUDA BENCH] mul broadcast-col  rows=" << rows
              << "  cols=" << cols << "\n";

    // a[i][0] = i*1+1,  b[i][j] = (i*cols+j)*1+1
    // expected c[i][j] = a[i][0] * b[i][j]
    auto expected_val = [&](int r, int c2)
    {
        return (static_cast<float>(r) * 1.f + 1.f) *
               (static_cast<float>(r * cols + c2) * 1.f + 1.f);
    };

    // CPU
    Tensor a_cpu({rows, 1}), b_cpu({rows, cols});
    fill_pattern(a_cpu.get_tensor_unrolled(), rows, 1.f);
    fill_pattern(b_cpu.get_tensor_unrolled(), rows * cols, 1.f);

    Tensor c_cpu({rows, cols});
    double cpu_ms = time_ms([&]
                            { c_cpu = mul(a_cpu, b_cpu); });

    for (auto [r, c2] : std::vector<std::pair<int, int>>{{0, 0}, {0, cols - 1}, {rows / 2, cols / 2}, {rows - 1, 0}, {rows - 1, cols - 1}})
        ASSERT_TRUE(approx_equal(c_cpu.get({r, c2}), expected_val(r, c2), 1e-2f),
                    "cpu broadcast-col mul [" + std::to_string(r) + "][" + std::to_string(c2) + "]");

    // CUDA
    Tensor a_cuda({rows, 1}), b_cuda({rows, cols});
    fill_pattern(a_cuda.get_tensor_unrolled(), rows, 1.f);
    fill_pattern(b_cuda.get_tensor_unrolled(), rows * cols, 1.f);
    a_cuda.set_device("cuda");
    b_cuda.set_device("cuda");

    mul(a_cuda, b_cuda);     // warmup
    cudaDeviceSynchronize(); // flush before timing

    Tensor c_cuda({rows, cols});
    double cuda_ms = time_ms([&]
                             {
                                 c_cuda = mul(a_cuda, b_cuda);
                                 cudaDeviceSynchronize(); // wait for kernel to finish
                             });

    for (auto [r, c2] : std::vector<std::pair<int, int>>{{0, 0}, {0, cols - 1}, {rows / 2, cols / 2}, {rows - 1, 0}, {rows - 1, cols - 1}})
        ASSERT_TRUE(approx_equal(c_cuda.get({r, c2}), expected_val(r, c2), 1e-2f),
                    "cuda broadcast-col mul [" + std::to_string(r) + "][" + std::to_string(c2) + "]");

    std::cout << "[CUDA BENCH] mul broadcast-col  rows=" << rows
              << "  cols=" << cols
              << "  cpu=" << cpu_ms << " ms"
              << "  cuda=" << cuda_ms << " ms"
              << "  speedup=" << (cpu_ms / cuda_ms) << "x\n";
}

#endif // USE_CUDA

// ── main ─────────────────────────────────────────────────────────────────────

int main()
{
    std::cout << "=== add tests ===\n";
    test_add_basic();
    test_add_2d();
    test_add_zeros();
    test_add_shape_mismatch();

    std::cout << "\n=== mul tests ===\n";
    test_mul_basic();
    test_mul_2d();
    test_mul_by_one();
    test_mul_by_zero();
    test_mul_shape_mismatch();

#ifdef USE_CUDA
    std::cout << "\n=== CUDA correctness (small tensors) ===\n";
    test_add_mul_cuda_contiguous();
#endif

    std::cout << "\n=== large tensor test ===\n";
    test_add_mul_large_tensor();

    std::cout << "\n=== broadcast tests: scalar {1} op {N} ===\n";
    test_add_broadcast_scalar();
    test_mul_broadcast_scalar();
    test_add_broadcast_scalar_reversed();
    test_mul_broadcast_scalar_reversed();

    std::cout << "\n=== broadcast tests: row {1,N} op {M,N} ===\n";
    test_add_broadcast_row();
    test_mul_broadcast_row();

    std::cout << "\n=== broadcast tests: col {M,1} op {M,N} ===\n";
    test_add_broadcast_col();
    test_mul_broadcast_col();

    std::cout << "\n=== broadcast tests: outer {M,1} op {1,N} ===\n";
    test_add_broadcast_outer();
    test_mul_broadcast_outer();

    std::cout << "\n=== broadcast tests: leading-dim {N} op {M,N} ===\n";
    test_add_broadcast_leading_dim();
    test_mul_broadcast_leading_dim();

    std::cout << "\n=== broadcast tests: incompatible shapes ===\n";
    test_add_broadcast_incompatible();
    test_mul_broadcast_incompatible();
    test_add_broadcast_incompatible_leading();
    test_mul_broadcast_incompatible_leading();

    std::cout << "\n=== broadcast tests: identity / edge cases ===\n";
    test_add_broadcast_with_zeros();
    test_mul_broadcast_with_ones();

    std::cout << "\n=== CPU benchmarks ===\n";
    bench_add(1024);
    bench_add(1 << 20);
    bench_mul(1024);
    bench_mul(1 << 20);
    bench_add_broadcast_row(1024, 1024);
    bench_mul_broadcast_col(1024, 1024);

#ifdef USE_CUDA
    std::cout << "\n=== CUDA vs CPU timing (3060-tuned sizes) ===\n";
    bench_add_cpu_vs_cuda_1d(1 << 20);               // 1 M elements  (~4 MB)
    bench_mul_cpu_vs_cuda_1d(1 << 20);               // 1 M elements  (~4 MB)
    bench_add_broadcast_row_cpu_vs_cuda(1024, 1024); // 1 M elements  (~4 MB)
    bench_mul_broadcast_col_cpu_vs_cuda(1024, 1024); // 1 M elements  (~4 MB)
#endif

    std::cout << "\n=== summary ===\n";
    std::cout << "passed: " << passed << "  failed: " << failed << "\n";

    return failed == 0 ? 0 : 1;
}