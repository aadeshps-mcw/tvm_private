import tvm
from tvm import te
from tvm.script import tir as T
import tvm.testing

def test_arithmetic_ops():
    print("=========================================")
    print("Test 1: Testing Arithmetic Operations (te)")
    print("=========================================")
    
    n = te.var("n")
    A = te.placeholder((n,), name="A", dtype="float32")
    B = te.placeholder((n,), name="B", dtype="float32")
    C = te.compute(A.shape, lambda i: (A[i] + B[i]) * (A[i] - 1.0) / 2.0, name="C")
    
    s = te.create_schedule(C.op)
    ir_mod = tvm.lower(s, [A, B, C], simple_mode=True)

    print("\n--- Triggering JSON Codegen ---")
    target = tvm.target.Target("json")
    mod = tvm.build(s, [A, B, C], target=target)
    
    output_file = "arithmetic_structure.json"
    mod.save(output_file)
    print(f"Saved Arithmetic Module to: {output_file}")


def test_comparison_ops():
    print("\n=========================================")
    print("Test 2: Testing Comparison Operations (TIR)")
    print("=========================================")

    @T.prim_func
    def kernel_eq(x: T.int32, y: T.int32, out: T.Buffer((1,), "bool")):
        out[0] = x == y

    @T.prim_func
    def kernel_lt(x: T.int32, out: T.Buffer((1,), "bool")):
        out[0] = x < 10
    
    mod = tvm.IRModule({"kernel_eq": kernel_eq, "kernel_lt": kernel_lt})

    target = tvm.target.Target("json")
    rt_mod = tvm.build(mod, target=target)
    
    output_file = "tir_structure.json"
    rt_mod.save(output_file)
    print(f"Saved Comparison Module to: {output_file}")


def test_control_flow_and_alloc():
    print("\n=========================================")
    print("Test 3: Control Flow & Allocations (IfThenElse, Allocate)")
    print("=========================================")

    @T.prim_func
    def kernel_control(A: T.Buffer((10,), "float32"), B: T.Buffer((10,), "float32")):
        temp = T.alloc_buffer((10,), "float32")
        
        for i in range(10):
            if A[i] > 0.0:
                temp[i] = A[i]
            else:
                temp[i] = T.float32(0.0)
                
        for i in range(10):
            B[i] = temp[i]

    mod = tvm.IRModule({"kernel_control": kernel_control})
    target = tvm.target.Target("json")
    rt_mod = tvm.build(mod, target=target)
    
    output_file = "control_flow.json"
    rt_mod.save(output_file)
    print(f"Saved Control Flow Module to: {output_file}")


def test_advanced_math_ops():
    print("\n=========================================")
    print("Test 4: Advanced Math (Call, Cast, Select)")
    print("=========================================")

    @T.prim_func
    def kernel_activation(x: T.Buffer((10,), "float32"), out: T.Buffer((10,), "int32")):
        for i in range(10):
            math_result = T.Select(x[i] > 0.0, T.exp(x[i]), T.float32(0.0))
            out[i] = T.cast(math_result, "int32")

    mod = tvm.IRModule({"kernel_activation": kernel_activation})
    target = tvm.target.Target("json")
    rt_mod = tvm.build(mod, target=target)
    
    output_file = "advanced_math.json"
    rt_mod.save(output_file)
    print(f"Saved Advanced Math Module to: {output_file}")

def test_dynamic_loops_and_constants():
    print("\n=========================================")
    print("Test 5: Dynamic Loops & Constants (While, AllocateConst)")
    print("=========================================")

    @T.prim_func
    def kernel_dynamic():
        i = T.alloc_buffer((1,), "int32")
        
        weights_ptr = T.allocate_const([10, 20, 30, 40], "int32", extents=[4])
        i[0] = 0
        
        while i[0] < 4:
            T.evaluate(T.isnullptr(weights_ptr))
            i[0] = i[0] + 1

    mod = tvm.IRModule({"kernel_dynamic": kernel_dynamic})
    target = tvm.target.Target("json")
    rt_mod = tvm.build(mod, target=target)
    
    output_file = "dynamic_loops.json"
    rt_mod.save(output_file)
    print(f"Saved Dynamic Loops Module to: {output_file}")

def test_simd_vectorization():
    print("\n=========================================")
    print("Test 6: SIMD Vectorization (Ramp, Broadcast, Shuffle)")
    print("=========================================")

    n = 16
    A = te.placeholder((n,), name="A", dtype="float32")

    B = te.compute(A.shape, lambda i: A[i] + 2.0, name="B")
    
    s = te.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=4)

    s[B].vectorize(xi)
    
    print("\n--- Triggering Vector Codegen ---")
    target = tvm.target.Target("json")
    mod = tvm.build(s, [A, B], target=target)
    
    output_file = "simd_vectorization.json"
    mod.save(output_file)
    print(f"Saved SIMD Vector Module to: {output_file}")


if __name__ == "__main__":
    if "json" not in tvm.target.Target.list_kinds():
        print("Error: 'json' target kind is not registered in target_kind.cc!")
        print("Ensure you modified target_kind.cc and recompiled TVM successfully.")
    else:
        test_arithmetic_ops()
        test_comparison_ops()
        test_control_flow_and_alloc()
        test_advanced_math_ops()
        test_dynamic_loops_and_constants()
        test_simd_vectorization()
        print("\nAll codegen smoke tests completed successfully! Check the generated .json files.")