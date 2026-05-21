import torch
import torch.nn as nn
import numpy as np

import tvm
from tvm import relay
from tvm.contrib import graph_executor

@tvm.tir.transform.prim_func_pass(opt_level=0)
def print_tir(f, mod, ctx):
    print("======================================")
    print("Generated TIR for function:", f.attrs.get("global_symbol", "main"))
    print("======================================")
    print(f)
    print("======================================\n")
    return f

class PowModel(nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.exponent = exponent

    def forward(self, x):
        return torch.pow(x, self.exponent)


def run_test(exponent):
    print(f"\n\n================= TEST: x^{exponent} =================")

    model = PowModel(exponent).eval()
    x = torch.randn(3, 4)

    pt_out = model(x).detach().numpy()


    traced = torch.jit.trace(model, (x,))
    traced.eval()

    input_shapes = [("x", x.shape)]
    mod, params = relay.frontend.from_pytorch(traced, input_shapes)

    # print("\n=== Relay IR ===")
    # print(mod)

    with tvm.transform.PassContext(
        opt_level=3,
        config={
            "tir.add_lower_pass": [
                (0, tvm.tir.transform.SimplifyPow()), 
                (1, print_tir),
            ]
        }
    ):
        lib = relay.build(mod, target="llvm", params=params)


    dev = tvm.cpu()
    m = graph_executor.GraphModule(lib["default"](dev))

    m.set_input("x", tvm.nd.array(x.numpy()))
    m.run()

    tvm_out = m.get_output(0).asnumpy()

    print("\nPyTorch Output:\n", pt_out)
    print("\nTVM Output:\n", tvm_out)
    print("\nMax Diff =", np.max(np.abs(pt_out - tvm_out)))


if __name__ == "__main__":
    run_test(0.0)
    run_test(1.0)
    run_test(-1.0)
    run_test(2.0)
    run_test(0.5)
    run_test(3.0)
    run_test(-0.5)
