/**
 * tests/test_matmul.cpp
 *
 * CTest-compatible tests for mat_mul().
 * Returns EXIT_SUCCESS (0) when all assertions pass,
 * EXIT_FAILURE (1) on the first failure so CTest marks the test as FAILED.
 *
 * No external test framework required — mirrors the style of
 * test_tensor.cpp and test_ops.cpp already in the project.
 *
 * Tensor API used (from Tensor.h):
 *   Tensor(std::vector<int> shape)          — allocates, values zero-initialised
 *   void set(const std::vector<int>&, float) — write one element
 *   float get(const std::vector<int>&) const — read one element
 *   const std::vector<int>& shape() const    — returns shape vector
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "Tensor.h"
#include "autograd/ops/Ops.h"   // mat_mul declaration

// ─────────────────────────────────────────────────────────────────────────────
//  Tiny assertion helpers
// ─────────────────────────────────────────────────────────────────────────────

static int g_passed = 0;
static int g_failed = 0;

static bool near_eq(float a, float b, float tol = 1e-5f)
{
    return std::fabs(a - b) <= tol;
}

#define ASSERT_NEAR(actual, expected, tol, msg)                                \
    do {                                                                       \
        float _a = static_cast<float>(actual);                                 \
        float _e = static_cast<float>(expected);                               \
        if (!near_eq(_a, _e, (tol))) {                                         \
            std::cerr << "[FAIL] " << (msg)                                    \
                      << "  got=" << _a << "  expected=" << _e << "\n";        \
            ++g_failed;                                                        \
        } else {                                                               \
            ++g_passed;                                                        \
        }                                                                      \
    } while (0)

#define ASSERT_THROWS(expr, msg)                                               \
    do {                                                                       \
        bool _threw = false;                                                   \
        try { (void)(expr); }                                                  \
        catch (const std::exception &) { _threw = true; }                     \
        catch (...) { _threw = true; }                                         \
        if (!_threw) {                                                         \
            std::cerr << "[FAIL] " << (msg) << " — expected exception\n";     \
            ++g_failed;                                                        \
        } else {                                                               \
            ++g_passed;                                                        \
        }                                                                      \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
//  Helper: construct a Tensor from a flat row-major data vector + shape.
//
//  Uses the real API:
//    Tensor(std::vector<int> shape)           — constructor
//    void set(const std::vector<int>&, float) — element writer
// ─────────────────────────────────────────────────────────────────────────────

static Tensor make_tensor(const std::vector<float> &data,
                           const std::vector<int>   &shape)
{
    Tensor t(shape);

    // Walk every multi-index in row-major order and call set().
    int rows = shape[0];
    int cols = shape[1];
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            t.set({i, j}, data[static_cast<size_t>(i * cols + j)]);

    return t;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Shorthand for reading an element — uses the real API: get({i, j})
// ─────────────────────────────────────────────────────────────────────────────

static float at(const Tensor &t, int i, int j)
{
    return t.get({i, j});
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test cases
// ─────────────────────────────────────────────────────────────────────────────

/**
 * test_matmul_square_2x2
 *
 * A = [[1, 2],   B = [[5, 6],   C = A @ B = [[19, 22],
 *      [3, 4]]        [7, 8]]                [43, 50]]
 */
static void test_matmul_square_2x2()
{
    std::cout << "[TEST] mat_mul — square 2x2\n";

    Tensor A = make_tensor({1.f, 2.f,
                            3.f, 4.f}, {2, 2});

    Tensor B = make_tensor({5.f, 6.f,
                            7.f, 8.f}, {2, 2});

    Tensor C = mat_mul(A, B);

    ASSERT_NEAR(at(C,0,0), 19.f, 1e-5f, "C[0,0]");
    ASSERT_NEAR(at(C,0,1), 22.f, 1e-5f, "C[0,1]");
    ASSERT_NEAR(at(C,1,0), 43.f, 1e-5f, "C[1,0]");
    ASSERT_NEAR(at(C,1,1), 50.f, 1e-5f, "C[1,1]");
}

/**
 * test_matmul_non_square
 *
 * A (2x3) @ B (3x2) -> C (2x2)
 *
 * A = [[1, 2, 3],   B = [[ 7,  8],   C = [[ 58,  64],
 *      [4, 5, 6]]        [ 9, 10],        [139, 154]]
 *                        [11, 12]]
 */
static void test_matmul_non_square()
{
    std::cout << "[TEST] mat_mul — non-square 2x3 @ 3x2\n";

    Tensor A = make_tensor({1.f, 2.f,  3.f,
                            4.f, 5.f,  6.f}, {2, 3});

    Tensor B = make_tensor({ 7.f,  8.f,
                              9.f, 10.f,
                             11.f, 12.f}, {3, 2});

    Tensor C = mat_mul(A, B);

    ASSERT_NEAR(at(C,0,0),  58.f, 1e-5f, "C[0,0]");
    ASSERT_NEAR(at(C,0,1),  64.f, 1e-5f, "C[0,1]");
    ASSERT_NEAR(at(C,1,0), 139.f, 1e-5f, "C[1,0]");
    ASSERT_NEAR(at(C,1,1), 154.f, 1e-5f, "C[1,1]");
}

/**
 * test_matmul_row_times_col  (dot product)
 *
 * A (1x3) @ B (3x1) -> scalar in a (1x1) tensor
 * [1, 2, 3] . [4, 5, 6]^T = 32
 */
static void test_matmul_row_times_col()
{
    std::cout << "[TEST] mat_mul — row x col (dot product, 1x3 @ 3x1)\n";

    Tensor A = make_tensor({1.f, 2.f, 3.f}, {1, 3});
    Tensor B = make_tensor({4.f, 5.f, 6.f}, {3, 1});

    Tensor C = mat_mul(A, B);

    ASSERT_NEAR(at(C,0,0), 32.f, 1e-5f, "dot product");
}

/**
 * test_matmul_identity
 *
 * A @ I == A  for any A.
 *
 * A = [[2, 3],   I = [[1, 0],
 *      [4, 5]]        [0, 1]]
 */
static void test_matmul_identity()
{
    std::cout << "[TEST] mat_mul — multiply by identity\n";

    Tensor A = make_tensor({2.f, 3.f,
                            4.f, 5.f}, {2, 2});

    Tensor I = make_tensor({1.f, 0.f,
                            0.f, 1.f}, {2, 2});

    Tensor C = mat_mul(A, I);

    ASSERT_NEAR(at(C,0,0), 2.f, 1e-5f, "C[0,0]");
    ASSERT_NEAR(at(C,0,1), 3.f, 1e-5f, "C[0,1]");
    ASSERT_NEAR(at(C,1,0), 4.f, 1e-5f, "C[1,0]");
    ASSERT_NEAR(at(C,1,1), 5.f, 1e-5f, "C[1,1]");
}

/**
 * test_matmul_zero_matrix
 *
 * A @ 0 == 0  (all elements of result are zero).
 */
static void test_matmul_zero_matrix()
{
    std::cout << "[TEST] mat_mul — multiply by zero matrix\n";

    Tensor A = make_tensor({1.f, 2.f,
                            3.f, 4.f}, {2, 2});

    Tensor Z = make_tensor({0.f, 0.f,
                            0.f, 0.f}, {2, 2});

    Tensor C = mat_mul(A, Z);

    ASSERT_NEAR(at(C,0,0), 0.f, 1e-5f, "C[0,0]");
    ASSERT_NEAR(at(C,0,1), 0.f, 1e-5f, "C[0,1]");
    ASSERT_NEAR(at(C,1,0), 0.f, 1e-5f, "C[1,0]");
    ASSERT_NEAR(at(C,1,1), 0.f, 1e-5f, "C[1,1]");
}

/**
 * test_matmul_output_shape
 *
 * A (3x4) @ B (4x5) -> C must have shape [3, 5].
 */
static void test_matmul_output_shape()
{
    std::cout << "[TEST] mat_mul — output shape 3x4 @ 4x5 -> 3x5\n";

    // Values are irrelevant; fill with 1s via set() through make_tensor.
    Tensor A = make_tensor(std::vector<float>(12, 1.f), {3, 4});
    Tensor B = make_tensor(std::vector<float>(20, 1.f), {4, 5});

    Tensor C = mat_mul(A, B);

    const std::vector<int> &sh = C.shape();
    bool ok = (sh.size() == 2 && sh[0] == 3 && sh[1] == 5);
    if (!ok) {
        std::cerr << "[FAIL] output shape: expected [3, 5], got ["
                  << (sh.size() > 0 ? sh[0] : -1) << ", "
                  << (sh.size() > 1 ? sh[1] : -1) << "]\n";
        ++g_failed;
    } else {
        ++g_passed;
    }
}

/**
 * test_matmul_incompatible_shapes
 *
 * A[2x3] @ B[4x2] must throw because inner dims don't match (3 != 4).
 */
static void test_matmul_incompatible_shapes()
{
    std::cout << "[TEST] mat_mul — incompatible shapes must throw\n";

    Tensor A = make_tensor(std::vector<float>(6, 1.f), {2, 3});
    Tensor B = make_tensor(std::vector<float>(8, 1.f), {4, 2});

    ASSERT_THROWS(mat_mul(A, B), "mat_mul({2,3}, {4,2}) should throw");
}

/**
 * test_matmul_large_values
 *
 * Checks there is no overflow / precision collapse for moderately large floats.
 *
 * A = [[100, 200],   B = [[300, 400],
 *      [300, 400]]        [500, 600]]
 *
 * C[0,0] = 100*300 + 200*500 = 130000
 * C[0,1] = 100*400 + 200*600 = 160000
 * C[1,0] = 300*300 + 400*500 = 290000
 * C[1,1] = 300*400 + 400*600 = 360000
 */
static void test_matmul_large_values()
{
    std::cout << "[TEST] mat_mul — large values (no overflow)\n";

    Tensor A = make_tensor({100.f, 200.f,
                            300.f, 400.f}, {2, 2});

    Tensor B = make_tensor({300.f, 400.f,
                            500.f, 600.f}, {2, 2});

    Tensor C = mat_mul(A, B);

    ASSERT_NEAR(at(C,0,0), 130000.f, 1.f, "C[0,0]");
    ASSERT_NEAR(at(C,0,1), 160000.f, 1.f, "C[0,1]");
    ASSERT_NEAR(at(C,1,0), 290000.f, 1.f, "C[1,0]");
    ASSERT_NEAR(at(C,1,1), 360000.f, 1.f, "C[1,1]");
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main()
{
    test_matmul_square_2x2();
    test_matmul_non_square();
    test_matmul_row_times_col();
    test_matmul_identity();
    test_matmul_zero_matrix();
    test_matmul_output_shape();
    test_matmul_incompatible_shapes();
    test_matmul_large_values();

    std::cout << "\n--------------------------------------\n";
    std::cout << "Results: " << g_passed << " passed, "
              << g_failed  << " failed\n";

    return g_failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}