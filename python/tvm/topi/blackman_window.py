import tvm
from tvm import te, tir
import math

@tvm.target.generic_func
def blackman_window(n, periodic=True, dtype="float32"):
    if n == 1:
        return te.compute(
            (1,),
            lambda i: tir.const(1.0, dtype),
            name="blackman_window"
        )

    denom = n if periodic else n - 1
    two_pi = 2 * math.pi
    four_pi = 4 * math.pi

    def compute(i):
        i_f = tir.Cast(dtype, i)
        denom_f = tir.Cast(dtype, denom)
        return (
            tir.const(0.42, dtype)
            - tir.const(0.5, dtype)
              * te.cos((two_pi * i_f) / denom_f)
            + tir.const(0.08, dtype)
              * te.cos((four_pi * i_f) / denom_f)
        )

    return te.compute((n,), compute, name="blackman_window")