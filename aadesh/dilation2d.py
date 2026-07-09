import tvm
from tvm import relay
import tvm
print("USING TVM FROM:", tvm.__file__)

def main():
    print("1. Creating TVM Relay graph for dilation2d...")

    data_shape = (1, 64, 32, 32)
    weight_shape = (64, 3, 3)


    data = relay.var("data", shape=data_shape, dtype="float32")
    weight = relay.var("weight", shape=weight_shape, dtype="float32")

    out = relay.image.dilation2d(
        data,
        weight,
        strides=(1, 1),
        padding=(1, 1),
        dilations=(1, 1),
        data_layout="NCHW",
        kernel_layout="IHW"
    )

    mod = tvm.IRModule.from_expr(out)

    print("\n" + "=" * 80)
    print("=== LOWERING GRAPH TO LOW-LEVEL TIR LOOP FUNCTIONS ===")
    print("=" * 80)

    @tvm.tir.transform.prim_func_pass(opt_level=0)
    def print_tir_pass(prim_func, ir_mod, pass_ctx):
        func_name = prim_func.attrs.get("global_symbol", "") if prim_func.attrs else ""
        
        if "dilation2d" in str(func_name):
            print(f"\n[TIR Function Found: {func_name}]")
            print("-" * 60)
            print(prim_func.script())
            print("-" * 60)
            
        return prim_func

    target = "llvm"
    with tvm.transform.PassContext(
        opt_level=3, 
        config={"tir.add_lower_pass": [(3, print_tir_pass)]}
    ):
        print("Compiling model and extracting TIR...")
        compiled_lib = relay.build(mod, target=target)

if __name__ == "__main__":
    main()
