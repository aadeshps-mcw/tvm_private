"""
Run a PyTorch model through TVM Relay with DNNL BYOC offload.

Ops matched by a registered DNNL pattern (conv2d, conv2d+relu,
conv2d+bias+relu, dense, etc. -- see pattern_table() in
python/tvm/relay/op/contrib/dnnl.py) get pulled out into their own
subgraph and executed via oneDNN at runtime.

Everything else stays as plain Relay ops in @main and is compiled
normally by TVM's LLVM backend -- that's your CPU fallback. No manual
op-splitting is needed; partition_for_dnnl() decides this for you based
on the registered pattern table.

Prerequisite: TVM must be built with USE_DNNL=ON (the JSON runtime
variant -- i.e. src/runtime/contrib/dnnl/dnnl_json_runtime.cc compiled
in), not just USE_DNNL=OFF/legacy BLAS support.
"""

import numpy as np
import torch
import torchvision.models as models

import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay.op.contrib import dnnl


def summarize_partitioning(mod):
    """Print which ops ended up on DNNL vs which stayed on LLVM/CPU."""
    dnnl_subgraphs = 0
    cpu_ops = set()
    for gvar, func in mod.functions.items():
        is_dnnl = bool(func.attrs) and func.attrs.get("Compiler") == "dnnl"
        if is_dnnl:
            dnnl_subgraphs += 1
            continue

        def visit(expr):
            if isinstance(expr, relay.Call) and isinstance(expr.op, tvm.ir.Op):
                cpu_ops.add(expr.op.name)

        relay.analysis.post_order_visit(func, visit)

    print(f"\nDNNL-offloaded subgraphs: {dnnl_subgraphs}")
    print(f"Ops still running on LLVM/CPU fallback: {sorted(cpu_ops)}\n")


def main():
    # 1. Load and trace a PyTorch model.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
    input_shape = (1, 3, 224, 224)
    example_input = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, example_input).eval()

    # 2. Import into Relay.
    input_name = "input0"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    # 3. Confirm DNNL codegen is compiled in TVM.
    if not tvm.get_global_func("relay.ext.dnnl", True):
        raise RuntimeError(
            "DNNL codegen not found in this TVM build. "
            "Rebuild with USE_DNNL=ON (JSON runtime) in config.cmake."
        )

    # ------------------------------------------------------------------
    # 4. Partition graph.
    # ------------------------------------------------------------------
    # Now calling the function directly from the dnnl module
    mod = dnnl.partition_for_dnnl(mod, params=params)
    
    summarize_partitioning(mod)

    # --- SAVE THE GRAPH ---
    output_file = "partitioned_graph.txt"
    with open(output_file, "w") as f:
        f.write(mod.astext())
    print(f"Full partitioned graph written to {output_file}\n")
    # -----------------------------------------

    # 5. Build for CPU.
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    # 6. Execute.
    dev = tvm.cpu(0)
    rt_mod = graph_executor.GraphModule(lib["default"](dev))

    data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
    rt_mod.set_input(input_name, data)
    rt_mod.run()
    output = rt_mod.get_output(0).numpy()

    print("Output shape:", output.shape)
    print("Output[0][:5]:", output[0][:5])


if __name__ == "__main__":
    main()
