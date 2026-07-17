"""
Simple JSON-dump test for the SimplifyClip / LegalizeQnnOpForDnnl fix,
using a small CUSTOM graph (no external model file needed) that is built to
actually match the legalizer's pattern (plain `add` for bias, real `clip`
node) -- unlike the real MobileNet TFLite model, which uses `nn.bias_add`
and never reaches this part of the pattern at all.

This dumps:
  - <stage>.json   -- tvm.ir.save_json() raw structured dump
  - <stage>.relay  -- human-readable astext() dump

for two scenarios:
  A) clip PRESENT in the input graph
  B) clip ABSENT (simulating what SimplifyClip would leave behind)

...at two pipeline points each: right after building the graph, and after
running legalize_qnn_for_dnnl(). Compare A vs B post-legalization to see
whether the fix makes both converge to the same legalized output.

Usage:
    python dump_json_custom.py
Then inspect e.g.:
    cat out/B_no_clip__after_legalize.json | python -m json.tool | less
    diff out/A_with_clip__after_legalize.relay out/B_no_clip__after_legalize.relay
"""
import json
import os

import numpy as np
import tvm
from tvm import relay


OUT_DIR = "out_json_dump"


def build_graph(include_clip):
    """conv2d -> +bias(plain add) -> requantize -> [clip] -> cast"""
    data_shape = (1, 4, 8, 8)
    wgh_shape = (4, 4, 3, 3)

    np.random.seed(0)
    wgh_np = np.random.randint(-10, 10, size=wgh_shape).astype("int8")
    bias_np = np.random.randint(-5, 5, size=(4, 1, 1)).astype("int32")

    src = relay.var("src", shape=data_shape, dtype="uint8")
    wgh = relay.const(wgh_np)
    bias = relay.const(bias_np)

    conv = relay.qnn.op.conv2d(
        src, wgh,
        input_zero_point=relay.const(0, "int32"),
        kernel_zero_point=relay.const(0, "int32"),
        input_scale=relay.const(0.5, "float32"),
        kernel_scale=relay.const(0.5, "float32"),
        padding=(1, 1), channels=4, kernel_size=(3, 3),
    )
    # Plain `add`, NOT nn.bias_add -- this is what the legalizer pattern
    # actually matches (is_op("add")(self.root, self.bias)).
    biased = relay.op.add(conv, bias)

    rq = relay.qnn.op.requantize(
        biased,
        input_scale=relay.const(0.25, "float32"),
        input_zero_point=relay.const(0, "int32"),
        output_scale=relay.const(0.5, "float32"),
        output_zero_point=relay.const(0, "int32"),
        out_dtype="int32",
    )

    node = rq
    if include_clip:
        node = relay.op.clip(node, a_min=0.0, a_max=255.0)

    casted = relay.op.cast(node, dtype="uint8")
    func = relay.Function(relay.analysis.free_vars(casted), casted)
    mod = tvm.IRModule.from_expr(func)
    return relay.transform.InferType()(mod)


def dump(mod, name):
    os.makedirs(OUT_DIR, exist_ok=True)

    # JSON dump
    json_str = tvm.ir.save_json(mod)
    json_path = os.path.join(OUT_DIR, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump(json.loads(json_str), f, indent=2)

    # Human-readable dump
    text_path = os.path.join(OUT_DIR, f"{name}.relay")
    with open(text_path, "w") as f:
        f.write(mod.astext(show_meta_data=False))

    print(f"  wrote {json_path}  ({os.path.getsize(json_path)} bytes)")
    print(f"  wrote {text_path}")

    # Quick summary printed to console too
    print(f"  --- {name}.relay ---")
    print(mod.astext(show_meta_data=False))
    print()


if __name__ == "__main__":
    from tvm.relay.op.contrib.dnnl import legalize_qnn_for_dnnl

    print("=== Scenario A: clip PRESENT in input ===")
    mod_a = build_graph(include_clip=True)
    dump(mod_a, "A_with_clip__before_legalize")
    mod_a_legalized = legalize_qnn_for_dnnl(mod_a)
    dump(mod_a_legalized, "A_with_clip__after_legalize")

    print("=== Scenario B: clip ABSENT (simulating SimplifyClip having stripped it) ===")
    mod_b = build_graph(include_clip=False)
    dump(mod_b, "B_no_clip__before_legalize")
    mod_b_legalized = legalize_qnn_for_dnnl(mod_b)
    dump(mod_b_legalized, "B_no_clip__after_legalize")

    print("=" * 70)
    print("Now compare:")
    print(f"  diff {OUT_DIR}/A_with_clip__after_legalize.relay {OUT_DIR}/B_no_clip__after_legalize.relay")
    print()
    print("If the fix is applied (pat.optional(is_op('clip')) in dnnl.py):")
    print("  -> both A and B should show the SAME fully-legalized graph")
    print("     (cast/add/multiply/clip/multiply/add/cast chain)")
    print()
    print("If the fix is NOT applied (still is_op('clip')(pat), hard-required):")
    print("  -> A will show the legalized chain")
    print("  -> B will show the ORIGINAL untouched qnn.requantize/cast, unlegalized")
