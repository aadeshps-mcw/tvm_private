import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor


def run_benchmark_case(dev, target, case_idx, config, num_runs=5):
    print(f"\n" + "=" * 60)
    tier = config.get("tier", "")
    print(f"CASE {case_idx} [{tier}]: {config['description']}")
    print(f"Data: {config['data_shape']}, Kernel: {config['weight_shape']}")
    print(
        f"Strides: {config['strides']}, Pad: {config['padding']}, Dilation: {config['dilations']}"
    )
    print(
        f"data_layout={config.get('data_layout', 'NCHW')}, "
        f"kernel_layout={config.get('kernel_layout', 'IHW')}"
    )
    print("=" * 60)

    data_shape = config["data_shape"]
    weight_shape = config["weight_shape"]

    np.random.seed(0)
    data_np = np.random.uniform(size=data_shape).astype("float32")
    weight_np = np.random.uniform(size=weight_shape).astype("float32")

    data = relay.var("data", shape=data_shape, dtype="float32")
    weight = relay.var("weight", shape=weight_shape, dtype="float32")

    out = relay.image.dilation2d(
        data,
        weight,
        strides=config["strides"],
        padding=config["padding"],
        dilations=config["dilations"],
        data_layout=config.get("data_layout", "NCHW"),
        kernel_layout=config.get("kernel_layout", "IHW"),
    )
    mod = tvm.IRModule.from_expr(out)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)

    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input("data", data_np)
    module.set_input("weight", weight_np)

    print(f"\n--- Profiling Results ({num_runs} Macro Runs) ---")
    run_medians = []

    for r in range(num_runs):
        # time_evaluator runs it 20 times (number) and repeats that 10 times internally
        timer = module.module.time_evaluator("run", dev, number=20, repeat=10)
        results = np.array(timer().results) * 1000

        med = np.median(results)
        run_medians.append(med)
        print(f"  Run {r+1}/{num_runs} Median Time: {med:.4f} ms")

    overall_avg = np.mean(run_medians)
    overall_std = np.std(run_medians)

    print(f"\n>> Final Average over {num_runs} runs: {overall_avg:.4f} ms")
    print(f">> Macro-Run Std Dev:            {overall_std:.4f} ms")

    return overall_avg


def main():
    target = "llvm"
    dev = tvm.cpu(0)
    num_runs = 5

    # ------------------------------------------------------------------
    # Test cases grouped by priority tier:
    #
    # TIER1 = realistic, common-case shapes -> optimize hardest for these
    # TIER2 = plausible but secondary -> should still be reasonably fast
    # TIER3 = boundary / correctness cases -> must be correct, not a perf
    #         priority (rare in practice: degenerate kernels, atrous rates)
    # ------------------------------------------------------------------
    test_cases = [
        # ---------------- TIER 1: core / most representative ----------------
        {
            "tier": "TIER1",
            "description": "Standard 3x3 Dilation (Baseline, NCHW)",
            "data_shape": (1, 64, 128, 128),
            "weight_shape": (64, 3, 3),
            "strides": (1, 1),
            "padding": (1, 1),
            "dilations": (1, 1),
            "data_layout": "NCHW",
            "kernel_layout": "IHW",
        },
        {
            "tier": "TIER1",
            "description": "Standard 3x3 Dilation (Baseline, NHWC)",
            # NHWC is what the TF frontend actually emits for Dilation2D;
            # kernel_layout must be HWI (weight shape KH,KW,C) in this case.
            "data_shape": (1, 128, 128, 64),
            "weight_shape": (3, 3, 64),
            "strides": (1, 1),
            "padding": (1, 1),
            "dilations": (1, 1),
            "data_layout": "NHWC",
            "kernel_layout": "HWI",
        },
        {
            "tier": "TIER1",
            "description": "Full-Res Grayscale Preprocessing (document/OCR-style)",
            "data_shape": (1, 1, 1024, 1024),
            "weight_shape": (1, 5, 5),
            "strides": (1, 1),
            "padding": (2, 2),
            "dilations": (1, 1),
        },
        {
            "tier": "TIER1",
            "description": "Large Spatial, Moderate Channels (mid-res feature-map morphology)",
            "data_shape": (1, 16, 512, 512),
            "weight_shape": (16, 5, 5),
            "strides": (1, 1),
            "padding": (2, 2),
            "dilations": (1, 1),
        },
        {
            "tier": "TIER1",
            "description": "Mid-Network Feature Map (higher channels, smaller spatial)",
            "data_shape": (1, 128, 56, 56),
            "weight_shape": (128, 3, 3),
            "strides": (1, 1),
            "padding": (1, 1),
            "dilations": (1, 1),
        },
        {
            "tier": "TIER1",
            "description": "Odd (non-power-of-2) Image Size",
            "data_shape": (1, 64, 127, 255),
            "weight_shape": (64, 3, 3),
            "strides": (1, 1),
            "padding": (1, 1),
            "dilations": (1, 1),
        },
        {
            "tier": "TIER1",
            "description": "Batched Inference (Batch Size 8)",
            "data_shape": (8, 64, 128, 128),
            "weight_shape": (64, 3, 3),
            "strides": (1, 1),
            "padding": (1, 1),
            "dilations": (1, 1),
        },
        # ---------------- TIER 2: secondary but plausible ----------------
        {
            "tier": "TIER2",
            "description": "Strided Dilation (Stride=2, downsampling)",
            "data_shape": (1, 64, 128, 128),
            "weight_shape": (64, 3, 3),
            "strides": (2, 2),
            "padding": (1, 1),
            "dilations": (1, 1),
        },
        {
            "tier": "TIER2",
            "description": "Heavy Channel Depth (upper realistic bound)",
            "data_shape": (1, 256, 64, 64),
            "weight_shape": (256, 3, 3),
            "strides": (1, 1),
            "padding": (1, 1),
            "dilations": (1, 1),
        },
        {
            "tier": "TIER2",
            "description": "5x5 Kernel (standalone, isolates kernel-size effect)",
            "data_shape": (1, 64, 128, 128),
            "weight_shape": (64, 5, 5),
            "strides": (1, 1),
            "padding": (2, 2),
            "dilations": (1, 1),
        },
        {
            "tier": "TIER2",
            "description": "7x7 Kernel (realistic upper bound on SE size)",
            "data_shape": (1, 64, 128, 128),
            "weight_shape": (64, 7, 7),
            "strides": (1, 1),
            "padding": (3, 3),
            "dilations": (1, 1),
        },
        {
            "tier": "TIER2",
            "description": "Tiny Input (deep feature map late in network)",
            "data_shape": (1, 64, 8, 8),
            "weight_shape": (64, 3, 3),
            "strides": (1, 1),
            "padding": (1, 1),
            "dilations": (1, 1),
        },
        # ---------------- TIER 3: boundary / correctness cases ----------------
        {
            "tier": "TIER3",
            "description": "1x1 Kernel (degenerate case)",
            "data_shape": (1, 64, 128, 128),
            "weight_shape": (64, 1, 1),
            "strides": (1, 1),
            "padding": (0, 0),
            "dilations": (1, 1),
        },
        {
            "tier": "TIER3",
            "description": "Atrous Structuring Element (Dilation Rate=2)",
            "data_shape": (1, 64, 128, 128),
            "weight_shape": (64, 3, 3),
            "strides": (1, 1),
            "padding": (2, 2),
            "dilations": (2, 2),
        },
    ]

    summary_results = {}

    for i, config in enumerate(test_cases, 1):
        avg_time = run_benchmark_case(dev, target, i, config, num_runs=num_runs)
        summary_results[f"Case {i} [{config.get('tier','')}]: {config['description']}"] = avg_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, t in summary_results.items():
        print(f"{name:65s} {t:.4f} ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
