"""
Tests for LegalizeQnnOpForDnnl / SimplifyClip interaction.

Goals:
  1. Confirm whether the DNNL QNN legalizer pattern ignores the actual
     a_min/a_max on the matched "clip" node (it currently hardcodes 0,255
     in the callback regardless of what was matched).
  2. Confirm the legalizer still matches correctly when the clip node has
     already been removed by SimplifyClip (i.e. validate the `.optional()`
     pattern fix).
  3. Regression-test that legalized output is numerically identical whether
     or not SimplifyClip has already stripped the redundant clip.

Run with:
    pytest -v test_dnnl_qnn_clip_legalize.py
"""
import numpy as np
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing.temp_op_attr import TempOpAttr

try:
    # Import path may vary slightly across TVM versions.
    from tvm.relay.op.contrib.dnnl import legalize_qnn_for_dnnl
except ImportError:
    from tvm.relay.op.contrib.dnnl.dnnl import legalize_qnn_for_dnnl


def build_qnn_conv_graph(
    clip_bounds=(0.0, 255.0), include_clip=True, out_dtype="uint8", wgh_np=None, bias_np=None
):
    """
    Builds: qnn.conv2d -> add(bias) -> qnn.requantize -> [clip] -> cast

    Set include_clip=False to simulate what the graph looks like *after*
    SimplifyClip has already stripped a redundant clip.
    Set clip_bounds to something other than (0,255) to test whether the
    legalizer respects or ignores the real bounds.

    IMPORTANT: when comparing two graphs built by separate calls to this
    function (e.g. via structural_equal), pass the SAME wgh_np/bias_np
    arrays to both calls. structural_equal compares constant *values*, not
    just shapes -- leaving these unseeded/regenerated per call means two
    "otherwise identical" graphs will differ only in random constant data
    and spuriously fail comparison.
    """
    data_shape = (1, 4, 8, 8)
    wgh_shape = (4, 4, 3, 3)

    if wgh_np is None:
        wgh_np = np.random.randint(-10, 10, size=wgh_shape).astype("int8")
    if bias_np is None:
        bias_np = np.random.randint(-5, 5, size=(4, 1, 1)).astype("int32")

    src = relay.var("src", shape=data_shape, dtype="uint8")
    wgh = relay.const(wgh_np)
    # The legalizer's pattern specifically matches plain "add" (not
    # nn.bias_add), so the bias constant itself must already be shaped to
    # broadcast correctly against the channel axis of NCHW output, i.e.
    # (C, 1, 1) rather than (C,).
    bias = relay.const(bias_np)

    conv = relay.qnn.op.conv2d(
        src,
        wgh,
        input_zero_point=relay.const(0, "int32"),
        kernel_zero_point=relay.const(0, "int32"),
        input_scale=relay.const(0.5, "float32"),
        kernel_scale=relay.const(0.5, "float32"),
        padding=(1, 1),
        channels=4,
        kernel_size=(3, 3),
    )
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
        node = relay.op.clip(node, a_min=clip_bounds[0], a_max=clip_bounds[1])

    casted = relay.op.cast(node, dtype=out_dtype)

    return relay.Function(relay.analysis.free_vars(casted), casted)


def legalize(func):
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = legalize_qnn_for_dnnl(mod)
    return mod


def test_legalizer_matches_with_default_clip_bounds():
    """Sanity check: standard 0,255 clip legalizes without error."""
    func = build_qnn_conv_graph(clip_bounds=(0.0, 255.0), include_clip=True)
    mod = legalize(func)
    text = mod.astext()
    assert "clip" in text, "Expected legalized graph to (re)introduce a clip op"


def test_legalizer_ignores_nonstandard_clip_bounds():
    """
    Documents current (possibly buggy) behavior: even if the source clip's
    a_min/a_max are NOT (0, 255), the legalizer's pattern match (is_op("clip")
    with no has_attr constraint) still fires, and the callback unconditionally
    re-emits clip(gr, 0, 255) -- discarding the original bounds.

    If this assertion ever starts failing, it likely means someone added a
    has_attr({"a_min": 0.0, "a_max": 255.0}) constraint to the pattern, which
    would be the "more correct" fix -- update this test accordingly.
    """
    func = build_qnn_conv_graph(clip_bounds=(-50.0, 50.0), include_clip=True)
    mod = legalize(func)
    text = mod.astext()

    # Confirm the graph doesn't retain the original -50/50 bounds anywhere,
    # i.e. they were silently discarded/replaced by the hardcoded 0,255 in
    # the callback (relay.op.clip(gr, 0, 255)).
    assert "-50" not in text and "50f" not in text, (
        "Unexpected: original non-standard clip bounds survived legalization. "
        "This may mean the pattern/callback behavior has changed."
    )


def test_legalizer_matches_without_clip_node_present():
    """
    This is the key regression test for the `.optional()` pattern fix.

    Simulates a graph where SimplifyClip has already run and removed the
    (redundant, 0-255) clip node before LegalizeQnnOpForDnnl sees it.
    Without the `.optional()` fix, the DFPattern requires a literal `clip`
    call and this pattern will FAIL to match -- the qnn.conv2d/add/requantize
    chain will be left un-legalized (no DNNL-compatible rewrite applied).
    """
    func = build_qnn_conv_graph(include_clip=False)
    mod = legalize(func)
    text = mod.astext()

    # If legalization fired, we expect to see the DNNL-compatible expansion
    # markers: an explicit cast to float32 and the o_scl multiply pattern,
    # rather than the untouched qnn.requantize/qnn.conv2d ops.
    assert "qnn.requantize" not in text, (
        "Legalizer did not fire when clip node was absent -- pattern is "
        "still requiring a literal clip. Check that `.optional()` was "
        "applied to both is_op(\"clip\")(pat) sites in LegalizeQnnOpForDnnl."
    )


def test_legalized_output_identical_with_and_without_redundant_clip():
    """
    Numerically compares legalization output between:
      (a) graph WITH a standard 0,255 clip present
      (b) same graph with that clip already stripped (as SimplifyClip would)

    Both should legalize to functionally identical graphs, since the clip
    was redundant (uint8 output range already implies 0-255 saturation).
    """
    wgh_np = np.random.randint(-10, 10, size=(4, 4, 3, 3)).astype("int8")
    bias_np = np.random.randint(-5, 5, size=(4, 1, 1)).astype("int32")

    func_with_clip = build_qnn_conv_graph(
        include_clip=True, clip_bounds=(0.0, 255.0), wgh_np=wgh_np, bias_np=bias_np
    )
    func_without_clip = build_qnn_conv_graph(
        include_clip=False, wgh_np=wgh_np, bias_np=bias_np
    )

    mod_a = legalize(func_with_clip)
    mod_b = legalize(func_without_clip)

    # Structural equality check (alpha-equivalence) is the strict version;
    # if constant folding order differs slightly, fall back to comparing
    # astext() modulo variable naming, or execute both and diff outputs.
    assert tvm.ir.structural_equal(mod_a["main"], mod_b["main"]), (
        "Legalized graphs differ depending on whether the redundant clip "
        "was present before legalization ran -- SimplifyClip and "
        "LegalizeQnnOpForDnnl are not properly decoupled."
    )


def test_end_to_end_simplify_then_legalize_matches_legalize_then_simplify():
    """
    Full pipeline order-independence check: running SimplifyExpr (which
    contains SimplifyClip) before vs. after LegalizeQnnOpForDnnl should
    produce equivalent final graphs, once the DNNL legalizer no longer
    depends on the literal clip node's presence.
    """
    func = build_qnn_conv_graph(include_clip=True, clip_bounds=(0.0, 255.0))
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    # Order A: simplify first, then legalize
    mod_a = relay.transform.InferType()(mod)
    mod_a = relay.transform.SimplifyExpr()(mod_a)
    mod_a = legalize_qnn_for_dnnl(mod_a)

    # Order B: legalize first, then simplify
    mod_b = legalize_qnn_for_dnnl(mod)
    mod_b = relay.transform.InferType()(mod_b)
    mod_b = relay.transform.SimplifyExpr()(mod_b)

    # structural_equal is a strict syntactic check -- it will report a
    # difference even when SimplifyExpr reorders algebraically-equivalent
    # ops differently depending on pass order (e.g. add-then-multiply vs.
    # multiply-then-add). What we actually care about is that both orderings
    # produce numerically equivalent results, so compare outputs instead.
    np.random.seed(0)
    data_np = np.random.randint(0, 255, size=(1, 4, 8, 8)).astype("uint8")

    def run(mod, data):
        ex = relay.create_executor("vm", mod=mod, device=tvm.cpu(), target="llvm")
        return ex.evaluate()(data).numpy()

    out_a = run(mod_a, data_np)
    out_b = run(mod_b, data_np)

    np.testing.assert_array_equal(
        out_a,
        out_b,
        err_msg=(
            "Pass ordering affects final legalized graph's numerical output -- "
            "SimplifyClip and LegalizeQnnOpForDnnl still have an ordering dependency."
        ),
    )


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))