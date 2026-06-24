// test_device.cpp
// Tests for CPU <-> GPU memory consistency via Tensor::set_device / Storage::set_device.
// Follows the same ASSERT / ASSERT_THROWS conventions as the existing test suite.
//
// Prerequisites:
//   - A CUDA-capable GPU must be present.
//   - Link with -lcudart (e.g. g++ test_device.cpp Tensor.cpp Storage.cpp -lcudart -o test_device).
//   - All GPU reads are done through the same public API (get / set / get_tensor_unrolled)
//     so tests are device-agnostic: they only care that values are consistent.

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "Tensor.h"
#include "utils.h"   // increment_index

using namespace std;

// ─────────────────────────────────────────────
//  Re-use the same harness from the main suite
// ─────────────────────────────────────────────

static int passed = 0;
static int failed = 0;

#define ASSERT(cond, msg)                                               \
    do {                                                                \
        if (!(cond)) {                                                  \
            cerr << "  FAIL [" << __func__ << "] " << msg << "\n";     \
            ++failed;                                                   \
        } else {                                                        \
            ++passed;                                                   \
        }                                                               \
    } while (0)

#define ASSERT_THROWS(expr, msg)                                        \
    do {                                                                \
        bool threw = false;                                             \
        try { (expr); }                                                 \
        catch (...) { threw = true; }                                   \
        if (!threw) {                                                   \
            cerr << "  FAIL [" << __func__ << "] expected exception: " \
                 << msg << "\n";                                        \
            ++failed;                                                   \
        } else {                                                        \
            ++passed;                                                   \
        }                                                               \
    } while (0)

static bool near(float a, float b) { return fabs(a - b) < 1e-6f; }

// Fill tensor with 0, 1, 2, … in logical row-major order (mirrors main suite)
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

// Read flat element i by iterating logical indices (mirrors main suite)
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

// Total number of elements for a given shape
static int numel(const vector<int> &shape)
{
    int n = 1;
    for (auto s : shape) n *= s;
    return n;
}

// ─────────────────────────────────────────────
//  CPU -> GPU: values survive the transfer
// ─────────────────────────────────────────────

// After moving to GPU, every element read back through get() must equal the
// value that was written on the CPU.
void test_cpu_to_gpu_values_preserved()
{
    Tensor t({3, 4});
    fill_rowmajor(t);           // write on CPU: 0..11

    t.set_device("cuda");       // migrate to GPU

    bool ok = true;
    for (int i = 0; i < numel(t.shape()); ++i)
        if (!near(get_flat(t, i), (float)i)) ok = false;

    ASSERT(ok, "all values must be intact after CPU->GPU transfer");
}

// ─────────────────────────────────────────────
//  GPU -> CPU: values survive the transfer
// ─────────────────────────────────────────────

// Write on CPU, push to GPU, pull back to CPU, then read — still correct.
void test_gpu_to_cpu_values_preserved()
{
    Tensor t({2, 3});
    fill_rowmajor(t);           // 0..5

    t.set_device("cuda");       // CPU -> GPU
    t.set_device("cpu");        // GPU -> CPU

    bool ok = true;
    for (int i = 0; i < numel(t.shape()); ++i)
        if (!near(get_flat(t, i), (float)i)) ok = false;

    ASSERT(ok, "all values must be intact after GPU->CPU round-trip");
}

// ─────────────────────────────────────────────
//  CPU -> GPU -> CPU round-trip (multiple dims)
// ─────────────────────────────────────────────

void test_round_trip_1d()
{
    Tensor t({8});
    fill_rowmajor(t);
    t.set_device("cuda");
    t.set_device("cpu");
    bool ok = true;
    for (int i = 0; i < 8; ++i)
        if (!near(get_flat(t, i), (float)i)) ok = false;
    ASSERT(ok, "1-D round-trip values must be intact");
}

void test_round_trip_2d()
{
    Tensor t({4, 5});
    fill_rowmajor(t);
    t.set_device("cuda");
    t.set_device("cpu");
    bool ok = true;
    for (int i = 0; i < numel(t.shape()); ++i)
        if (!near(get_flat(t, i), (float)i)) ok = false;
    ASSERT(ok, "2-D round-trip values must be intact");
}

void test_round_trip_3d()
{
    Tensor t({2, 3, 4});
    fill_rowmajor(t);
    t.set_device("cuda");
    t.set_device("cpu");
    bool ok = true;
    for (int i = 0; i < numel(t.shape()); ++i)
        if (!near(get_flat(t, i), (float)i)) ok = false;
    ASSERT(ok, "3-D round-trip values must be intact");
}

// ─────────────────────────────────────────────
//  Set on GPU, read on CPU
// ─────────────────────────────────────────────

// Write a value while the tensor lives on the GPU, bring it back to CPU,
// then verify the write was not lost.
void test_set_on_gpu_readable_on_cpu()
{
    Tensor t({3, 3});
    fill_rowmajor(t);           // 0..8 on CPU

    t.set_device("cuda");

    // Overwrite one element while on GPU
    t.set({1, 1}, 99.0f);

    t.set_device("cpu");        // bring back

    ASSERT(near(t.get({1, 1}), 99.0f),
           "value written on GPU must be visible after moving back to CPU");
}

// ─────────────────────────────────────────────
//  Set on CPU after round-trip, read on GPU
// ─────────────────────────────────────────────

void test_set_on_cpu_after_roundtrip_readable_on_gpu()
{
    Tensor t({2, 4});
    fill_rowmajor(t);

    t.set_device("cuda");
    t.set_device("cpu");

    // Overwrite on CPU after the round-trip
    t.set({0, 3}, 77.0f);

    t.set_device("cuda");       // push updated data to GPU

    ASSERT(near(t.get({0, 3}), 77.0f),
           "value written on CPU after round-trip must be visible on GPU");
}

// ─────────────────────────────────────────────
//  Shape / strides are device-independent
// ─────────────────────────────────────────────

// Migrating to GPU must not alter shape or strides.
void test_shape_strides_unchanged_on_gpu()
{
    Tensor t({3, 4});
    const auto shape_before   = t.shape();
    const auto strides_before = t.strides();

    t.set_device("cuda");

    bool shape_ok   = (t.shape()   == shape_before);
    bool strides_ok = (t.strides() == strides_before);

    ASSERT(shape_ok,   "shape must be unchanged after moving to GPU");
    ASSERT(strides_ok, "strides must be unchanged after moving to GPU");
}

// ─────────────────────────────────────────────
//  Idempotent: moving to the same device is a no-op
// ─────────────────────────────────────────────

void test_idempotent_cpu_to_cpu()
{
    Tensor t({2, 3});
    fill_rowmajor(t);
    t.set_device("cpu");        // already on CPU: no-op
    bool ok = true;
    for (int i = 0; i < numel(t.shape()); ++i)
        if (!near(get_flat(t, i), (float)i)) ok = false;
    ASSERT(ok, "CPU->CPU should be a no-op and leave data intact");
}

void test_idempotent_gpu_to_gpu()
{
    Tensor t({2, 3});
    fill_rowmajor(t);
    t.set_device("cuda");
    t.set_device("cuda");       // already on GPU: no-op
    t.set_device("cpu");        // bring back to verify
    bool ok = true;
    for (int i = 0; i < numel(t.shape()); ++i)
        if (!near(get_flat(t, i), (float)i)) ok = false;
    ASSERT(ok, "GPU->GPU should be a no-op and leave data intact");
}

// ─────────────────────────────────────────────
//  get_tensor_unrolled pointer consistency
// ─────────────────────────────────────────────

// After moving to GPU and back, the CPU-side raw pointer returned by
// get_tensor_unrolled() must still reflect the correct element values.
void test_raw_pointer_consistent_after_round_trip()
{
    Tensor t({6});
    fill_rowmajor(t);           // 0..5

    t.set_device("cuda");
    t.set_device("cpu");

    const float *ptr = t.get_tensor_unrolled();
    bool ok = true;
    for (int i = 0; i < 6; ++i)
        if (!near(ptr[i], (float)i)) ok = false;

    ASSERT(ok, "get_tensor_unrolled() must be consistent after GPU round-trip");
}

// ─────────────────────────────────────────────
//  Transposed view: values survive device move
// ─────────────────────────────────────────────

// A transposed view shares storage. Moving that storage to GPU and back must
// leave the logical transposed values intact.
void test_transposed_view_round_trip()
{
    Tensor t({2, 3});
    fill_rowmajor(t);           // 0..5 row-major
    Tensor tr = t.transpose(0, 1);  // {3,2}

    // Capture expected values before any device move
    vector<float> expected;
    {
        vector<int> idx(tr.shape().size(), 0);
        int total = numel(tr.shape());
        for (int i = 0; i < total; ++i)
        {
            expected.push_back(tr.get(idx));
            increment_index(idx, tr.shape());
        }
    }

    // Move the underlying storage through GPU and back via the original tensor
    t.set_device("cuda");
    t.set_device("cpu");

    // Verify transposed view still reads correctly
    bool ok = true;
    {
        vector<int> idx(tr.shape().size(), 0);
        int total = numel(tr.shape());
        for (int i = 0; i < total; ++i)
        {
            if (!near(tr.get(idx), expected[i])) ok = false;
            increment_index(idx, tr.shape());
        }
    }
    ASSERT(ok, "transposed view values must be intact after GPU round-trip");
}

// ─────────────────────────────────────────────
//  Invalid device string throws
// ─────────────────────────────────────────────

void test_invalid_device_throws()
{
    Tensor t({4});
    ASSERT_THROWS(t.set_device("tpu"), "unknown device string should throw");
    ASSERT_THROWS(t.set_device(""),    "empty device string should throw");
    ASSERT_THROWS(t.set_device("CPU"), "wrong case device string should throw");
}

// ─────────────────────────────────────────────
//  Multiple round-trips: data stays stable
// ─────────────────────────────────────────────

void test_multiple_round_trips()
{
    Tensor t({3, 3});
    fill_rowmajor(t);   // 0..8

    for (int i = 0; i < 3; ++i)
    {
        t.set_device("cuda");
        t.set_device("cpu");
    }

    bool ok = true;
    for (int i = 0; i < numel(t.shape()); ++i)
        if (!near(get_flat(t, i), (float)i)) ok = false;

    ASSERT(ok, "data must remain correct across three GPU round-trips");
}

// ─────────────────────────────────────────────
//  Large tensor: stress test for transfer
// ─────────────────────────────────────────────

void test_large_tensor_round_trip()
{
    // 256 × 256 = 65 536 elements – small enough to be fast, large enough to
    // catch any off-by-one or partial-copy bug.
    Tensor t({256, 256});
    fill_rowmajor(t);

    t.set_device("cuda");
    t.set_device("cpu");

    bool ok = true;
    for (int i = 0; i < numel(t.shape()); ++i)
        if (!near(get_flat(t, i), (float)i)) { ok = false; break; }

    ASSERT(ok, "large tensor (256x256) must survive GPU round-trip intact");
}

// ─────────────────────────────────────────────
//  Runner
// ─────────────────────────────────────────────

int main()
{
    cout << "=== device: CPU -> GPU value preservation ===\n";
    test_cpu_to_gpu_values_preserved();

    cout << "=== device: GPU -> CPU value preservation ===\n";
    test_gpu_to_cpu_values_preserved();

    cout << "=== device: round-trip by dimensionality ===\n";
    test_round_trip_1d();
    test_round_trip_2d();
    test_round_trip_3d();

    cout << "=== device: writes on one device visible on the other ===\n";
    test_set_on_gpu_readable_on_cpu();
    test_set_on_cpu_after_roundtrip_readable_on_gpu();

    cout << "=== device: metadata invariance ===\n";
    test_shape_strides_unchanged_on_gpu();

    cout << "=== device: idempotent transfers ===\n";
    test_idempotent_cpu_to_cpu();
    test_idempotent_gpu_to_gpu();

    cout << "=== device: raw pointer consistency ===\n";
    test_raw_pointer_consistent_after_round_trip();

    cout << "=== device: view (transpose) + device migration ===\n";
    test_transposed_view_round_trip();

    cout << "=== device: error handling ===\n";
    test_invalid_device_throws();

    cout << "=== device: stability under repeated transfers ===\n";
    test_multiple_round_trips();

    cout << "=== device: large tensor stress ===\n";
    test_large_tensor_round_trip();

    cout << "\n";
    if (failed == 0)
        cout << "All " << passed << " tests passed.\n";
    else
        cout << passed << " passed, " << failed << " FAILED.\n";

    return failed == 0 ? 0 : 1;
}