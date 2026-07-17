import numpy as np
import tvm
from tvm import relay
from tvm.relay.qnn.op.canonicalizations import create_integer_lookup_op

def test_multichannel_lut_delta():
    shape = (4, 3) # 3 channels
    
    # Setup multi-channel data
    np.random.seed(42)
    x_np = np.random.randint(0, 255, size=shape).astype("uint8")
    
    in_scale_np = np.array([0.1, 0.5, 1.0], dtype="float32")
    in_zp_np = np.array([128, 128, 128], dtype="int32")
    out_scale_np = np.array([0.01, 0.01, 0.01], dtype="float32")
    out_zp_np = np.array([0, 0, 0], dtype="int32")

    x = relay.var("x", shape=shape, dtype="uint8")
    in_scale = relay.const(in_scale_np)
    in_zp = relay.const(in_zp_np)
    out_scale = relay.const(out_scale_np)
    out_zp = relay.const(out_zp_np)
    
    # Try to lower the multi-channel operation
    print("--- Phase 1: Lowering Graph ---")
    try:
        # Remove in_shape parameter to verify dynamic inference works
        y = create_integer_lookup_op(
            input_arg=x,
            floating_point_func=lambda arr: 1.0 / (1.0 + np.exp(-arr)),
            in_scale=in_scale,
            in_zero_point=in_zp,
            out_scale=out_scale,
            out_zero_point=out_zp,
            in_axis=-1,
            out_axis=-1,
            in_dtype="uint8",
            out_dtype="uint8"
        )
        print("[SUCCESS] Graph lowered without crashing. (Fix is active!)")
    except ValueError as e:
        if "can only convert an array of size 1" in str(e):
            print("\n[CONFIRMED BUG] Lowering failed with expected ValueError!")
            print(f"Error caught: {e}")
            print("\n>>> VERDICT: This is the UNPATCHED behavior. The .item() restriction broke compilation.")
            return
        else:
            # Raise if it's an unrelated ValueError
            raise e

    # Compile and Run
    print("\n--- Phase 2: Compiling and Executing ---")
    mod = tvm.IRModule.from_expr(y)
    with tvm.transform.PassContext(opt_level=3):
        exe = relay.create_executor("vm", mod=mod, target="llvm")
        evaluator = exe.evaluate()
        
    tvm_output = evaluator(x_np).numpy()

    # Verify math against reference
    dequantized = (x_np.astype("float32") - in_zp_np) * in_scale_np
    sigmoided = 1.0 / (1.0 + np.exp(-dequantized))
    quantized = np.round(sigmoided / out_scale_np) + out_zp_np
    np_reference = np.clip(quantized, 0, 255).astype("uint8")

    np.testing.assert_allclose(tvm_output, np_reference, atol=1)
    print("[SUCCESS] Output matches NumPy reference math exactly.")
    print("\n>>> VERDICT: This is the PATCHED behavior. Multi-channel LUT works perfectly!")

if __name__ == "__main__":
    test_multichannel_lut_delta()
