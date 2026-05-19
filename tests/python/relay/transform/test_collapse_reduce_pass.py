import tvm
import tvm.relay as relay
import tvm.relay.transform as _transform

def run_pass(func):
    mod = tvm.IRModule.from_expr(func)
    print("IR BEFORE PASS:")
    print(mod)

    seq = tvm.transform.Sequential(
        [
            _transform.InferType(),
            _transform.CollapseReduceAxis0Pass(),
            _transform.InferType(),
        ]
    )
    with tvm.transform.PassContext(opt_level=0):
        mod = seq(mod)

    print("\n")
    print("IR AFTER PASS:")
    print(mod)

    return mod["main"]
# for types ir and comparison to be correct, run infertype on expected func and mod before comparison
def add_expected_to_mod(mod, expected_func):
    gv = relay.GlobalVar("expected")
    mod[gv] = expected_func
    mod = _transform.InferType()(mod)
    return mod["expected"]

def test_reduce_axis0_no_keepdims():
    shape = (2, 3, 4)
    data = relay.var("data", shape=shape, dtype="float32")

    # Original redue_sum
    out = relay.sum(data, axis=[0], keepdims=False)
    before_func = relay.Function([data], out)

    # reshape to reduce to reshape
    data_e = relay.var("data", shape=shape, dtype="float32")
    flat = relay.reshape(data_e, (2, 12))            # [2, 12]
    red = relay.sum(flat, axis=[0], keepdims=False)  # [12]
    restored = relay.reshape(red, (3, 4))            # [3, 4]

    expected_func = relay.Function([data_e], restored)
    result = run_pass(before_func)
    mod = tvm.IRModule.from_expr(result)
    expected = add_expected_to_mod(mod, expected_func)

    tvm.ir.assert_structural_equal(result, expected)
    print("PASS: test_reduce_axis0_no_keepdims")



def test_reduce_axis0_keepdims():
    shape = (2, 3, 4)
    data = relay.var("data", shape=shape, dtype="float32")

    out = relay.sum(data, axis=[0], keepdims=True)
    before_func = relay.Function([data], out)
    data_e = relay.var("data", shape=shape, dtype="float32")

    flat = relay.reshape(data_e, (2, 12))
    red = relay.sum(flat, axis=[0], keepdims=True)   # keepdims=True
    restored = relay.reshape(red, (1, 3, 4))

    expected_func = relay.Function([data_e], restored)
    result = run_pass(before_func)
    mod = tvm.IRModule.from_expr(result)
    expected = add_expected_to_mod(mod, expected_func)
    tvm.ir.assert_structural_equal(result, expected)
    print(" PASS: test_reduce_axis0_keepdims")


def test_reduce_non_axis0():
    shape = (2, 3, 4)
    data = relay.var("data", shape=shape, dtype="float32")

    # axis=1, should remain unchanged
    out = relay.sum(data, axis=[1], keepdims=False)
    before_func = relay.Function([data], out)
    result = run_pass(before_func)
    mod = tvm.IRModule.from_expr(result)
    expected = add_expected_to_mod(mod, before_func)
    tvm.ir.assert_structural_equal(result, expected)
    print(" PASS: test_reduce_non_axis0")

if __name__ == "__main__":
    test_reduce_axis0_no_keepdims()
    test_reduce_axis0_keepdims()
    test_reduce_non_axis0()
    print("\n Alln CollapseReduceAxis0Pass tests passed!")
