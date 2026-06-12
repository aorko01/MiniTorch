#include "Ops.h"
#include "Tensor.h"
#include <iostream>
#include <chrono>
#include <cassert>
#include <cmath>
#include <vector>

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

#define ASSERT_TRUE(cond, msg)                                      \
    do {                                                            \
        if (!(cond)) {                                              \
            std::cerr << "[FAIL] " << msg << "\n";                 \
            ++failed;                                               \
        } else {                                                    \
            std::cout << "[PASS] " << msg << "\n";                 \
            ++passed;                                               \
        }                                                           \
    } while (0)

// ── add tests ────────────────────────────────────────────────────────────────

void test_add_basic()
{
    // 1-D: [3]
    Tensor a({3}), b({3});
    a.set({0}, 1.0f); a.set({1}, 2.0f); a.set({2}, 3.0f);
    b.set({0}, 4.0f); b.set({1}, 5.0f); b.set({2}, 6.0f);

    Tensor c = add(a, b);

    ASSERT_TRUE(approx_equal(c.get({0}), 5.0f), "add basic [0] == 5");
    ASSERT_TRUE(approx_equal(c.get({1}), 7.0f), "add basic [1] == 7");
    ASSERT_TRUE(approx_equal(c.get({2}), 9.0f), "add basic [2] == 9");
}

void test_add_2d()
{
    // 2-D: [2][2]
    Tensor a({2, 2}), b({2, 2});
    a.set({0,0}, 1.f); a.set({0,1}, 2.f);
    a.set({1,0}, 3.f); a.set({1,1}, 4.f);

    b.set({0,0}, 10.f); b.set({0,1}, 20.f);
    b.set({1,0}, 30.f); b.set({1,1}, 40.f);

    Tensor c = add(a, b);

    ASSERT_TRUE(approx_equal(c.get({0,0}), 11.f), "add 2d [0][0] == 11");
    ASSERT_TRUE(approx_equal(c.get({0,1}), 22.f), "add 2d [0][1] == 22");
    ASSERT_TRUE(approx_equal(c.get({1,0}), 33.f), "add 2d [1][0] == 33");
    ASSERT_TRUE(approx_equal(c.get({1,1}), 44.f), "add 2d [1][1] == 44");
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
    try { add(a, b); }
    catch (const std::runtime_error &) { threw = true; }
    ASSERT_TRUE(threw, "add shape mismatch throws");
}

// ── mul tests ────────────────────────────────────────────────────────────────

void test_mul_basic()
{
    Tensor a({3}), b({3});
    a.set({0}, 2.0f); a.set({1}, 3.0f); a.set({2}, 4.0f);
    b.set({0}, 5.0f); b.set({1}, 6.0f); b.set({2}, 7.0f);

    Tensor c = mul(a, b);

    ASSERT_TRUE(approx_equal(c.get({0}), 10.f), "mul basic [0] == 10");
    ASSERT_TRUE(approx_equal(c.get({1}), 18.f), "mul basic [1] == 18");
    ASSERT_TRUE(approx_equal(c.get({2}), 28.f), "mul basic [2] == 28");
}

void test_mul_2d()
{
    Tensor a({2, 2}), b({2, 2});
    a.set({0,0}, 1.f); a.set({0,1}, 2.f);
    a.set({1,0}, 3.f); a.set({1,1}, 4.f);

    b.set({0,0}, 2.f); b.set({0,1}, 3.f);
    b.set({1,0}, 4.f); b.set({1,1}, 5.f);

    Tensor c = mul(a, b);

    ASSERT_TRUE(approx_equal(c.get({0,0}),  2.f), "mul 2d [0][0] == 2");
    ASSERT_TRUE(approx_equal(c.get({0,1}),  6.f), "mul 2d [0][1] == 6");
    ASSERT_TRUE(approx_equal(c.get({1,0}), 12.f), "mul 2d [1][0] == 12");
    ASSERT_TRUE(approx_equal(c.get({1,1}), 20.f), "mul 2d [1][1] == 20");
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
    try { mul(a, b); }
    catch (const std::runtime_error &) { threw = true; }
    ASSERT_TRUE(threw, "mul shape mismatch throws");
}

// ── timing benchmarks ────────────────────────────────────────────────────────

void bench_add(int size)
{
    std::vector<int> shape = {size};
    Tensor a(shape), b(shape);

    const float *ra = a.get_tensor_unrolled();
    const float *rb = b.get_tensor_unrolled();
    float *wa = const_cast<float *>(ra);
    float *wb = const_cast<float *>(rb);
    for (int i = 0; i < size; ++i) { wa[i] = static_cast<float>(i); wb[i] = 1.f; }

    double ms = time_ms([&]{ add(a, b); });
    std::cout << "[BENCH] add  size=" << size << "  time=" << ms << " ms\n";
}

void bench_mul(int size)
{
    std::vector<int> shape = {size};
    Tensor a(shape), b(shape);

    const float *ra = a.get_tensor_unrolled();
    float *wa = const_cast<float *>(ra);
    const float *rb = b.get_tensor_unrolled();
    float *wb = const_cast<float *>(rb);
    for (int i = 0; i < size; ++i) { wa[i] = static_cast<float>(i); wb[i] = 2.f; }

    double ms = time_ms([&]{ mul(a, b); });
    std::cout << "[BENCH] mul  size=" << size << "  time=" << ms << " ms\n";
}

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

    std::cout << "\n=== benchmarks ===\n";
    bench_add(1024);
    bench_add(1 << 20);   // ~1 M elements
    bench_mul(1024);
    bench_mul(1 << 20);

    std::cout << "\n=== summary ===\n";
    std::cout << "passed: " << passed << "  failed: " << failed << "\n";

    return failed == 0 ? 0 : 1;
}