import numpy as np
import tvm
from tvm import te, topi
import tvm.testing


def blackman_ref(n, periodic):
    if n == 1:
        return np.array([1.0], dtype="float32")
    denom = n if periodic else n - 1
    x = np.arange(n, dtype="float32")
    return (
        0.42
        - 0.5 * np.cos(2 * np.pi * x / denom)
        + 0.08 * np.cos(4 * np.pi * x / denom)
    ).astype("float32")


@tvm.testing.parametrize_targets("llvm")
def test_topi_blackman_window(target):
    for n in [1, 2, 16, 128]:
        for periodic in [True, False]:
            with tvm.target.Target(target):
                out = topi.blackman_window(n, periodic, "float32")
                s = te.create_schedule(out.op)

                func = tvm.build(s, [out], target)
                dev = tvm.device(target, 0)

                out_tvm = tvm.nd.empty((n,), "float32", dev)
                func(out_tvm)

            np.testing.assert_allclose(
                out_tvm.numpy(),
                blackman_ref(n, periodic),
                rtol=1e-5,
                atol=1e-6,
            )