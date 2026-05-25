import tvm
import tvm.testing
from tvm import te
import numpy as np

def apply_simplify_pow(stmt, params):
    mod = tvm.IRModule.from_expr(
        tvm.tir.PrimFunc(params, stmt).with_attr( #wraps into PrimFunc, which is then wrapped into IRModule
            "target", tvm.target.Target("llvm")
        )
    )
    mod = tvm.tir.transform.SimplifyPow()(mod)
    func = mod["main"]

    return func.body

def check_pow(exp, data):
    n = len(data)

    A = te.placeholder((n,), name="A", dtype="float32")
    C = te.compute((n,), lambda i: tvm.tir.pow(A[i], tvm.tir.const(exp, "float32")))
    s = te.create_schedule(C.op)

    with tvm.transform.PassContext(
        opt_level=3,
        config={
            "tir.add_lower_pass": [
                (0, tvm.tir.transform.SimplifyPow())
            ]
        },
    ):
        f = tvm.build(s, [A, C], "llvm")

    a_np = np.array(data, dtype="float32")
    a = tvm.nd.array(a_np)
    c = tvm.nd.array(np.zeros_like(a_np))

    f(a, c)

    if exp == 0.0:
        ref = np.ones_like(a_np)
    elif exp == 1.0:
        ref = a_np
    elif exp == -1.0:
        ref = 1.0 / a_np

    np.testing.assert_allclose(c.numpy(), ref, rtol=1e-5)

@tvm.testing.requires_llvm
def test_pow_zero():
    data = np.random.randn(10)
    check_pow(0.0, data)


@tvm.testing.requires_llvm
def test_pow_one():
    data = np.random.randn(10)
    check_pow(1.0, data)


@tvm.testing.requires_llvm
def test_pow_minus_one():
    data = np.random.randn(10)
    check_pow(-1.0, data)


if __name__ == "__main__":
    test_pow_zero()
    test_pow_one()
    test_pow_minus_one()