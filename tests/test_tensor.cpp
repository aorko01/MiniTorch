#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "Tensor.h"
#include "utils.h"

using namespace std;

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────

static int passed = 0;
static int failed = 0;

#define ASSERT(cond, msg)                                         \
    do {                                                          \
        if (!(cond)) {                                            \
            cerr << "  FAIL [" << __func__ << "] " << msg << "\n"; \
            ++failed;                                             \
        } else {                                                  \
            ++passed;                                             \
        }                                                         \
    } while (0)

#define ASSERT_THROWS(expr, msg)                                  \
    do {                                                          \
        bool threw = false;                                       \
        try { (expr); }                                           \
        catch (...) { threw = true; }                             \
        if (!threw) {                                             \
            cerr << "  FAIL [" << __func__ << "] expected exception: " << msg << "\n"; \
            ++failed;                                             \
        } else {                                                  \
            ++passed;                                             \
        }                                                         \
    } while (0)

static bool vec_eq(const vector<int> &a, const vector<int> &b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

static bool near(float a, float b) { return fabs(a - b) < 1e-6f; }

// Mirrors the private is_contiguous() logic using only the public API.
// A tensor is contiguous iff strides are descending powers matching shape.
static bool is_contiguous(const Tensor &t)
{
    const auto &shape   = t.shape();
    const auto &strides = t.strides();
    int expected = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i)
    {
        if (strides[i] != expected) return false;
        expected *= shape[i];
    }
    return true;
}

// Fill tensor with 0, 1, 2, … in logical row-major order
static void fill_rowmajor(Tensor &t)
{
    const auto &shape = t.shape();
    int n = 1;
    for (auto s : shape) n *= s;
    vector<int> idx(shape.size(), 0);
    for (int i = 0; i < n; ++i)
    {
        t.set(idx, (float)i);
        increment_index(idx, shape);
    }
}

// Read flat element i by iterating logical indices in order
static float get_flat(const Tensor &t, int flat_idx)
{
    const auto &shape = t.shape();
    int n = 1;
    for (auto s : shape) n *= s;
    vector<int> idx(shape.size(), 0);
    for (int i = 0; i < n; ++i)
    {
        if (i == flat_idx) return t.get(idx);
        increment_index(idx, shape);
    }
    return 0.0f;
}

// ─────────────────────────────────────────────
//  increment_index tests
// ─────────────────────────────────────────────

// Basic single-element carry: last dim increments first
void test_increment_basic()
{
    vector<int> idx = {0, 0};
    vector<int> shape = {2, 3};
    increment_index(idx, shape);
    ASSERT(vec_eq(idx, {0, 1}), "first increment should give {0,1}");
}

// Carry from last dim into previous dim
void test_increment_carry()
{
    vector<int> idx = {0, 2};     // last position in dim-1
    vector<int> shape = {2, 3};
    increment_index(idx, shape);
    ASSERT(vec_eq(idx, {1, 0}), "carry should give {1,0}");
}

// Full wrap-around: last index wraps all dims back to zero
void test_increment_full_wraparound()
{
    vector<int> idx = {1, 2};     // last valid index for shape {2,3}
    vector<int> shape = {2, 3};
    increment_index(idx, shape);
    ASSERT(vec_eq(idx, {0, 0}), "full wrap should give {0,0}");
}

// 1-D shape: straightforward increment
void test_increment_1d()
{
    vector<int> idx = {3};
    vector<int> shape = {5};
    increment_index(idx, shape);
    ASSERT(vec_eq(idx, {4}), "1-D increment should give {4}");
}

// 3-D multi-carry: {0,2,2} with shape {2,3,3} -> {1,0,0}
void test_increment_3d_carry()
{
    vector<int> idx = {0, 2, 2};
    vector<int> shape = {2, 3, 3};
    increment_index(idx, shape);
    ASSERT(vec_eq(idx, {1, 0, 0}), "3-D carry should give {1,0,0}");
}

// Enumerate all indices for a small shape; count should equal total elements
void test_increment_full_enumeration()
{
    vector<int> shape = {2, 3, 2};
    int total = 12;
    vector<int> idx(3, 0);
    int count = 0;
    for (int i = 0; i < total; ++i)
    {
        ++count;
        increment_index(idx, shape);
    }
    ASSERT(count == total, "full enumeration count should equal total elements");
    ASSERT(vec_eq(idx, {0, 0, 0}), "after full wrap index should be back to {0,0,0}");
}

// ─────────────────────────────────────────────
//  Tensor constructor / strides
// ─────────────────────────────────────────────

void test_strides_1d()
{
    Tensor t({5});
    ASSERT(vec_eq(t.strides(), {1}), "1-D strides should be {1}");
}

void test_strides_2d()
{
    Tensor t({3, 4});
    ASSERT(vec_eq(t.strides(), {4, 1}), "2-D strides for {3,4} should be {4,1}");
}

void test_strides_3d()
{
    Tensor t({3, 2, 2});
    ASSERT(vec_eq(t.strides(), {4, 2, 1}), "3-D strides for {3,2,2} should be {4,2,1}");
}

void test_strides_4d()
{
    Tensor t({2, 3, 4, 5});
    ASSERT(vec_eq(t.strides(), {60, 20, 5, 1}), "4-D strides should be {60,20,5,1}");
}

// ─────────────────────────────────────────────
//  get / set / location
// ─────────────────────────────────────────────

void test_get_set_basic()
{
    Tensor t({2, 3});
    t.set({1, 2}, 9.5f);
    ASSERT(near(t.get({1, 2}), 9.5f), "set then get should return same value");
}

void test_get_set_zero_index()
{
    Tensor t({3, 3});
    t.set({0, 0}, 42.0f);
    ASSERT(near(t.get({0, 0}), 42.0f), "set/get at {0,0} should work");
}

void test_get_set_all_elements()
{
    Tensor t({2, 2, 2});
    fill_rowmajor(t);
    // Verify every element independently
    int n = 8;
    vector<int> idx(3, 0);
    bool ok = true;
    for (int i = 0; i < n; ++i)
    {
        if (!near(t.get(idx), (float)i)) ok = false;
        increment_index(idx, t.shape());
    }
    ASSERT(ok, "all elements should match their fill value");
}

void test_get_bad_index_throws()
{
    Tensor t({2, 3});
    ASSERT_THROWS(t.get({0, 0, 0}), "wrong rank index should throw");
}

// ─────────────────────────────────────────────
//  is_contiguous
// ─────────────────────────────────────────────

void test_contiguous_fresh_tensor()
{
    Tensor t({3, 4});
    ASSERT(is_contiguous(t), "freshly constructed tensor must be contiguous");
}

void test_contiguous_after_transpose()
{
    Tensor t({3, 4});
    Tensor tr = t.transpose(0, 1);
    ASSERT(!is_contiguous(tr), "transposed tensor must not be contiguous");
}

void test_contiguous_1d()
{
    Tensor t({10});
    ASSERT(is_contiguous(t), "1-D tensor should be contiguous");
}

void test_contiguous_3d_fresh()
{
    Tensor t({2, 3, 4});
    ASSERT(is_contiguous(t), "fresh 3-D tensor should be contiguous");
}

// ─────────────────────────────────────────────
//  transpose
// ─────────────────────────────────────────────

void test_transpose_shape()
{
    Tensor t({2, 3});
    Tensor tr = t.transpose(0, 1);
    ASSERT(vec_eq(tr.shape(), {3, 2}), "transpose should swap shape dims");
}

void test_transpose_strides()
{
    Tensor t({2, 3});           // strides {3,1}
    Tensor tr = t.transpose(0, 1);
    ASSERT(vec_eq(tr.strides(), {1, 3}), "transpose should swap strides");
}

void test_transpose_values()
{
    Tensor t({2, 3});
    fill_rowmajor(t);           // 0..5
    Tensor tr = t.transpose(0, 1);
    bool ok = true;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            if (!near(tr.get({i, j}), t.get({j, i}))) ok = false;
    ASSERT(ok, "transposed values: tr(i,j) must equal t(j,i)");
}

void test_transpose_shares_storage()
{
    // Mutating original should be visible through the transposed view
    Tensor t({2, 3});
    fill_rowmajor(t);
    Tensor tr = t.transpose(0, 1);
    t.set({0, 1}, 99.0f);
    ASSERT(near(tr.get({1, 0}), 99.0f), "transpose should share storage with original");
}

void test_transpose_3d()
{
    Tensor t({2, 3, 4});
    Tensor tr = t.transpose(0, 2);
    ASSERT(vec_eq(tr.shape(), {4, 3, 2}), "3-D transpose(0,2) should give {4,3,2}");
}

void test_transpose_same_dim()
{
    Tensor t({3, 4});
    Tensor tr = t.transpose(0, 0);  // no-op
    ASSERT(vec_eq(tr.shape(), {3, 4}), "transposing same dim should be a no-op");
    ASSERT(vec_eq(tr.strides(), t.strides()), "no-op transpose strides unchanged");
}

void test_transpose_bad_dim_throws()
{
    Tensor t({2, 3});
    ASSERT_THROWS(t.transpose(0, 2), "out-of-bounds dim should throw");
    ASSERT_THROWS(t.transpose(-1, 0), "negative dim should throw");
}

// ─────────────────────────────────────────────
//  reshape – contiguous path
// ─────────────────────────────────────────────

void test_reshape_contiguous_shape()
{
    Tensor t({2, 3});
    Tensor r = t.reshape({3, 2});
    ASSERT(vec_eq(r.shape(), {3, 2}), "reshape should update shape");
}

void test_reshape_contiguous_strides()
{
    Tensor t({2, 3});
    Tensor r = t.reshape({3, 2});
    ASSERT(vec_eq(r.strides(), {2, 1}), "contiguous reshape strides should be row-major");
}

void test_reshape_contiguous_values()
{
    Tensor t({2, 3});
    fill_rowmajor(t);
    Tensor r = t.reshape({3, 2});
    bool ok = true;
    for (int i = 0; i < 6; ++i)
        if (!near(get_flat(r, i), (float)i)) ok = false;
    ASSERT(ok, "reshape should preserve logical element order");
}

void test_reshape_to_1d()
{
    Tensor t({2, 3});
    fill_rowmajor(t);
    Tensor r = t.reshape({6});
    ASSERT(vec_eq(r.shape(), {6}), "reshape to 1-D should give shape {6}");
    bool ok = true;
    for (int i = 0; i < 6; ++i)
        if (!near(get_flat(r, i), (float)i)) ok = false;
    ASSERT(ok, "reshape to 1-D values should be correct");
}

void test_reshape_from_1d()
{
    Tensor t({6});
    fill_rowmajor(t);
    Tensor r = t.reshape({2, 3});
    ASSERT(vec_eq(r.shape(), {2, 3}), "reshape from 1-D should give {2,3}");
}

void test_reshape_same_shape()
{
    Tensor t({3, 2});
    fill_rowmajor(t);
    Tensor r = t.reshape({3, 2});
    ASSERT(vec_eq(r.shape(), {3, 2}), "reshape to same shape should work");
    ASSERT(near(get_flat(r, 3), 3.0f), "values should be unchanged");
}

void test_reshape_incompatible_throws()
{
    Tensor t({2, 3});
    ASSERT_THROWS(t.reshape({4, 2}), "incompatible reshape should throw");
}

// ─────────────────────────────────────────────
//  reshape – non-contiguous path
// ─────────────────────────────────────────────

void test_reshape_noncontiguous_shape()
{
    Tensor t({2, 3});
    Tensor tr = t.transpose(0, 1);   // {3,2}, non-contiguous
    Tensor r = tr.reshape({6});
    ASSERT(vec_eq(r.shape(), {6}), "non-contiguous reshape should give {6}");
}

void test_reshape_noncontiguous_is_contiguous()
{
    Tensor t({2, 3});
    Tensor tr = t.transpose(0, 1);
    Tensor r = tr.reshape({6});
    ASSERT(is_contiguous(r), "result of non-contiguous reshape must be contiguous");
}

void test_reshape_noncontiguous_values()
{
    Tensor t({2, 3});
    fill_rowmajor(t);               // logical: 0..5 in row-major
    Tensor tr = t.transpose(0, 1);  // {3,2}: tr(i,j)=t(j,i)
    Tensor r = tr.reshape({6});

    // Build expected by iterating tr's logical indices
    vector<float> expected;
    vector<int> idx(tr.shape().size(), 0);
    int total = 1;
    for (auto s : tr.shape()) total *= s;
    for (int i = 0; i < total; ++i)
    {
        expected.push_back(tr.get(idx));
        increment_index(idx, tr.shape());
    }

    bool ok = true;
    for (int i = 0; i < total; ++i)
        if (!near(get_flat(r, i), expected[i])) ok = false;
    ASSERT(ok, "non-contiguous reshape values should match tr's logical order");
}

void test_reshape_noncontiguous_3d()
{
    Tensor t({2, 3, 4});
    fill_rowmajor(t);
    Tensor tr = t.transpose(0, 2);  // {4,3,2}, non-contiguous
    Tensor r = tr.reshape({24});
    ASSERT(vec_eq(r.shape(), {24}), "3-D non-contiguous reshape to {24}");
    ASSERT(is_contiguous(r), "result should be contiguous");
}

// ─────────────────────────────────────────────
//  Runner
// ─────────────────────────────────────────────

int main()
{
    cout << "=== increment_index ===\n";
    test_increment_basic();
    test_increment_carry();
    test_increment_full_wraparound();
    test_increment_1d();
    test_increment_3d_carry();
    test_increment_full_enumeration();

    cout << "=== strides ===\n";
    test_strides_1d();
    test_strides_2d();
    test_strides_3d();
    test_strides_4d();

    cout << "=== get / set ===\n";
    test_get_set_basic();
    test_get_set_zero_index();
    test_get_set_all_elements();
    test_get_bad_index_throws();

    cout << "=== is_contiguous ===\n";
    test_contiguous_fresh_tensor();
    test_contiguous_after_transpose();
    test_contiguous_1d();
    test_contiguous_3d_fresh();

    cout << "=== transpose ===\n";
    test_transpose_shape();
    test_transpose_strides();
    test_transpose_values();
    test_transpose_shares_storage();
    test_transpose_3d();
    test_transpose_same_dim();
    test_transpose_bad_dim_throws();

    cout << "=== reshape (contiguous) ===\n";
    test_reshape_contiguous_shape();
    test_reshape_contiguous_strides();
    test_reshape_contiguous_values();
    test_reshape_to_1d();
    test_reshape_from_1d();
    test_reshape_same_shape();
    test_reshape_incompatible_throws();

    cout << "=== reshape (non-contiguous) ===\n";
    test_reshape_noncontiguous_shape();
    test_reshape_noncontiguous_is_contiguous();
    test_reshape_noncontiguous_values();
    test_reshape_noncontiguous_3d();

    cout << "\n";
    if (failed == 0)
        cout << "All " << passed << " tests passed.\n";
    else
        cout << passed << " passed, " << failed << " FAILED.\n";

    return failed == 0 ? 0 : 1;
}