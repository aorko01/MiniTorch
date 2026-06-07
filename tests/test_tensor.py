import math
import minitorch_cpp as mt

# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────

passed = 0
failed = 0


def ASSERT(cond, msg):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print("  FAIL:", msg)


def ASSERT_THROWS(fn, msg):
    global passed, failed
    try:
        fn()
        failed += 1
        print("  FAIL: expected exception:", msg)
    except:
        passed += 1


def vec_eq(a, b):
    return list(a) == list(b)


def near(a, b):
    return abs(a - b) < 1e-6


def increment_index(idx, shape):
    # same logic as your C++ utils.h version
    for i in reversed(range(len(shape))):
        idx[i] += 1
        if idx[i] < shape[i]:
            return
        idx[i] = 0


def is_contiguous(t):
    expected = 1
    shape = t.shape()
    strides = t.strides()

    for i in reversed(range(len(shape))):
        if strides[i] != expected:
            return False
        expected *= shape[i]
    return True


def fill_rowmajor(t):
    shape = t.shape()
    n = 1
    for s in shape:
        n *= s

    idx = [0] * len(shape)
    for i in range(n):
        t.set(idx, float(i))
        increment_index(idx, shape)


def get_flat(t, flat_idx):
    shape = t.shape()
    n = 1
    for s in shape:
        n *= s

    idx = [0] * len(shape)
    for i in range(n):
        if i == flat_idx:
            return t.get(idx)
        increment_index(idx, shape)
    return 0.0


# ─────────────────────────────────────────────
# tests
# ─────────────────────────────────────────────

def test_strides():
    t = mt.Tensor([3, 2, 2])
    ASSERT(vec_eq(t.strides(), [4, 2, 1]), "strides mismatch")


def test_get_set():
    t = mt.Tensor([2, 3])
    t.set([1, 2], 9.5)
    ASSERT(near(t.get([1, 2]), 9.5), "get/set failed")


def test_transpose():
    t = mt.Tensor([2, 3])
    fill_rowmajor(t)

    tr = t.transpose(0, 1)

    ASSERT(vec_eq(tr.shape(), [3, 2]), "transpose shape")
    ASSERT(vec_eq(tr.strides(), [1, 3]), "transpose strides")

    ok = True
    for i in range(2):
        for j in range(3):
            if not near(tr.get([j, i]), t.get([i, j])):
                ok = False
    ASSERT(ok, "transpose values")


def test_transpose_view_sharing():
    t = mt.Tensor([2, 3])
    fill_rowmajor(t)

    tr = t.transpose(0, 1)
    t.set([0, 1], 99.0)

    ASSERT(near(tr.get([1, 0]), 99.0), "view sharing failed")


def test_reshape_contiguous():
    t = mt.Tensor([2, 3])
    fill_rowmajor(t)

    r = t.reshape([3, 2])

    ASSERT(vec_eq(r.shape(), [3, 2]), "reshape shape")

    ok = True
    for i in range(6):
        if not near(get_flat(r, i), float(i)):
            ok = False
    ASSERT(ok, "reshape values")


def test_reshape_1d():
    t = mt.Tensor([2, 3])
    fill_rowmajor(t)

    r = t.reshape([6])

    ok = True
    for i in range(6):
        if not near(get_flat(r, i), float(i)):
            ok = False
    ASSERT(ok, "reshape to 1d")


def test_reshape_incompatible():
    t = mt.Tensor([2, 3])
    ASSERT_THROWS(lambda: t.reshape([4, 2]), "bad reshape not caught")


def test_noncontiguous_reshape():
    t = mt.Tensor([2, 3])
    fill_rowmajor(t)

    tr = t.transpose(0, 1)
    r = tr.reshape([6])

    ASSERT(vec_eq(r.shape(), [6]), "noncontiguous reshape shape")


def test_all():
    test_strides()
    test_get_set()
    test_transpose()
    test_transpose_view_sharing()
    test_reshape_contiguous()
    test_reshape_1d()
    test_reshape_incompatible()
    test_noncontiguous_reshape()


# ─────────────────────────────────────────────
# runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_all()

    print("\nRESULTS")
    print("passed:", passed)
    print("failed:", failed)

    if failed > 0:
        exit(1)