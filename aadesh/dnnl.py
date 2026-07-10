import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor
from tvm.ir.instrument import PrintAfterAll

def run_dnnl_byoc():
    # 1. Define shapes
    data_shape = (1, 64, 56, 56) # NCHW
    weight_shape = (64, 64, 3, 3)
    
    data = relay.var("data", shape=data_shape, dtype="float32")
    weight = relay.var("weight", shape=weight_shape, dtype="float32")
    
    conv = relay.nn.conv2d(
        data, 
        weight, 
        padding=(1, 1), 
        channels=64, 
        kernel_size=(3, 3)
    )
    out = relay.nn.relu(conv)
    
    mod = tvm.IRModule.from_expr(out)

    # 2. Generate actual data (STRICTLY POSITIVE to prevent ReLU flattening to 0)
    np_data = np.random.uniform(0.1, 1.0, size=data_shape).astype("float32")
    np_weight = np.random.uniform(0.1, 1.0, size=weight_shape).astype("float32")

    # 3. Bind parameters so the partitioning pass tracks the weights natively
    params = {"weight": tvm.nd.array(np_weight)}
    mod["main"] = relay.build_module.bind_params_by_name(mod["main"], params)

    print("=== Original Module (With Bound Weights) ===")
    print(mod)

    # 4. Setup the BYOC partitioning passes for DNNL
    pattern_table = tvm.relay.op.contrib.get_pattern_table("dnnl")
    
    seq = tvm.transform.Sequential([
        relay.transform.MergeComposite(pattern_table),
        relay.transform.AnnotateTarget("dnnl"),
        relay.transform.MergeCompilerRegions(),
        relay.transform.PartitionGraph()
    ])

    # Partitioning under PassContext
    with tvm.transform.PassContext(opt_level=3, instruments=[PrintAfterAll()]):
        mod_partitioned = seq(mod)
        
    print("\n=== Partitioned Module ===")
    print(mod_partitioned)

    # 5. Compile the partitioned module
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod_partitioned, target=target)

    # 6. Execute using Graph Executor
    dev = tvm.cpu(0)
    module = graph_executor.GraphModule(lib["default"](dev))

    # Pass the variable runtime data
    module.set_input("data", np_data)
    
    print("\nExecuting model on DNNL backend...")
    module.run()

    # Get the output
    out_tensor = module.get_output(0).numpy()
    print("Execution successful! Output shape:", out_tensor.shape)
    print("Sample output values (Should be non-zero now):\n", out_tensor[0, 0, 0, :5])

if __name__ == "__main__":
    run_dnnl_byoc()