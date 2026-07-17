import tvm
from tvm import relay
import numpy as np
import tvm.testing

def test_fused_sum_aliasing_and_layout():
    """
    Tests that the DNNL JSON runtime correctly handles:
    1. Layout mismatches on the sum (skip) tensor.
    2. Aliasing hazards when the sum tensor is consumed by multiple nodes.
    """
    # Change channels to 256
    shape = (1, 256, 56, 56)
    data = relay.var("data", shape=shape, dtype="float32")
    weight1 = relay.var("weight1", shape=(256, 256, 3, 3), dtype="float32")
    weight2 = relay.var("weight2", shape=(256, 256, 3, 3), dtype="float32")

    skip_tensor = relay.nn.conv2d(data, weight1, padding=(1, 1), channels=256, kernel_size=(3, 3))
    conv_out = relay.nn.conv2d(skip_tensor, weight2, padding=(1, 1), channels=256, kernel_size=(3, 3))
    
    # THE FUSION TARGET: DNNL will fuse this Add into `conv_out` using a sum post-op.
    fused_add = relay.add(conv_out, skip_tensor)
    
    # If the runtime mutates skip_tensor in-place without cloning, 
    # this branch reads corrupted memory.
    hazard_branch = relay.add(skip_tensor, relay.const(1.0, dtype="float32"))
    
    # Output both branches to force TVM to compute the hazard branch
    out = relay.Tuple([fused_add, hazard_branch])
    func = relay.Function(relay.analysis.free_vars(out), out)
    mod = tvm.IRModule.from_expr(func)

    np.random.seed(42)
    inputs = {
        "data": np.random.uniform(-1, 1, shape).astype("float32"),
        "weight1": np.random.uniform(-1, 1, (256, 256, 3, 3)).astype("float32"),
        "weight2": np.random.uniform(-1, 1, (256, 256, 3, 3)).astype("float32"),
    }

    target_llvm = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib_llvm = relay.build(mod, target=target_llvm)
    
    rt_llvm = tvm.contrib.graph_executor.GraphModule(lib_llvm["default"](tvm.cpu()))
    rt_llvm.set_input(**inputs)
    rt_llvm.run()
    ref_fused_add = rt_llvm.get_output(0).numpy()
    ref_hazard = rt_llvm.get_output(1).numpy()

    
    seq = tvm.transform.Sequential([
        relay.transform.AnnotateTarget("dnnl"),
        relay.transform.MergeCompilerRegions(),
        relay.transform.PartitionGraph()
    ])
    
    with tvm.transform.PassContext(opt_level=3):
        mod_dnnl = seq(mod)
        lib_dnnl = relay.build(mod_dnnl, target=target_llvm)

    rt_dnnl = tvm.contrib.graph_executor.GraphModule(lib_dnnl["default"](tvm.cpu()))
    rt_dnnl.set_input(**inputs)
    rt_dnnl.run()
    dnnl_fused_add = rt_dnnl.get_output(0).numpy()
    dnnl_hazard = rt_dnnl.get_output(1).numpy()

    # If the layout is mismatched, this assertion will fail.
    tvm.testing.assert_allclose(dnnl_fused_add, ref_fused_add, rtol=1e-4, atol=1e-4)
    tvm.testing.assert_allclose(dnnl_hazard, ref_hazard, rtol=1e-4, atol=1e-4)
    
    print("Test passed! DNNL output matches LLVM reference exactly.")

if __name__ == "__main__":
    test_fused_sum_aliasing_and_layout()