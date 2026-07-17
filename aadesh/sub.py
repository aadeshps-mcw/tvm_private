"""
Structural test for the QnnSubtractCanonicalize fast path.

Numerical correctness tests (see test_qnn_subtract_fastpath.py) can't prove
the fast path actually fired -- the old, always-two-requantize code produces
identical numeric output, just via more (redundant) ops. To prove the fast
path is really being taken, we inspect the CANONICALIZED graph structure and
count qnn.requantize nodes directly.

Design: choose lhs_scale == rhs_scale (so the fast path is eligible), but
make BOTH differ from output_scale (so RequantizeOrUpcast is forced to emit
a real requantize step every time it's called, rather than silently
upcasting when params already match the target).

IMPORTANT: `relay.qnn.transform.CanonicalizeOps()` is recursive -- it does
not stop at lowering qnn.subtract. Any qnn.requantize node produced during
that lowering gets immediately lowered too, down into its primitive building
blocks (cast, fixed_point_multiply, add, clip). So `qnn.requantize` never
survives as a literal node in the final canonicalized graph -- counting it
directly always yields 0. Instead we count `fixed_point_multiply`, which is
a reliable 1:1 proxy: each requantize's core fixed-point scaling step lowers
to exactly one fixed_point_multiply call, regardless of how the rest of that
requantize gets expanded.

Expected fixed_point_multiply count for `Subtract(lhs, rhs)`, both quantized
identically, with a differing output scale:
  - BEFORE fix (always requantize both operands into output params first):
        2x fixed_point_multiply  (one per operand's requantize)
  - AFTER fix (fast path: subtract raw values, requantize once at the end):
        1x fixed_point_multiply  (only the final diff -> output conversion)

Run with:
    pytest -v test_qnn_subtract_structural.py
"""
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor


class OpCounter(ExprVisitor):
    """Counts Call nodes by op name across the whole expression."""

    def __init__(self):
        super().__init__()
        self.counts = {}

    def visit_call(self, call):
        op_name = None
        if isinstance(call.op, tvm.ir.Op):
            op_name = call.op.name
        elif hasattr(call.op, "name_hint"):
            # Could be a Function (e.g. after MergeComposite) -- not expected
            # here, but guard anyway.
            op_name = None
        if op_name:
            self.counts[op_name] = self.counts.get(op_name, 0) + 1
        super().visit_call(call)


def count_ops(expr):
    counter = OpCounter()
    counter.visit(expr)
    return counter.counts


def canonicalize_qnn_subtract(lhs_scale, lhs_zp, rhs_scale, rhs_zp, out_scale, out_zp, dtype="uint8"):
    shape = (2, 4)
    lhs = relay.var("lhs", shape=shape, dtype=dtype)
    rhs = relay.var("rhs", shape=shape, dtype=dtype)

    out = relay.qnn.op.subtract(
        lhs,
        rhs,
        lhs_scale=relay.const(lhs_scale, "float32"),
        lhs_zero_point=relay.const(lhs_zp, "int32"),
        rhs_scale=relay.const(rhs_scale, "float32"),
        rhs_zero_point=relay.const(rhs_zp, "int32"),
        output_scale=relay.const(out_scale, "float32"),
        output_zero_point=relay.const(out_zp, "int32"),
    )
    func = relay.Function([lhs, rhs], out)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    # Run ONLY qnn canonicalization -- deliberately avoid FoldConstant/
    # DeadCodeElimination/etc, since we want to inspect the raw structural
    # output of QnnSubtractCanonicalize itself, not whatever a later
    # optimization pass might further collapse it into.
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    mod = relay.transform.InferType()(mod)

    return mod["main"].body


def test_fast_path_emits_single_requantize():
    """
    lhs_scale == rhs_scale, both != out_scale -- fast path should fire and
    the canonicalized graph should contain exactly ONE qnn.requantize node.
    """
    body = canonicalize_qnn_subtract(
        lhs_scale=0.2, lhs_zp=0, rhs_scale=0.2, rhs_zp=0, out_scale=0.5, out_zp=5
    )
    counts = count_ops(body)

    assert counts.get("fixed_point_multiply", 0) == 1, (
        f"Expected exactly 1 fixed_point_multiply (requantize proxy) when lhs/rhs "
        f"share qnn params (fast path), got {counts.get('fixed_point_multiply', 0)}. "
        f"Full op counts: {counts}. This likely means the fast path in "
        f"QnnSubtractCanonicalize did not fire -- check that the IsEqualScalar guard "
        f"is being hit and that the fix is actually compiled into this build."
    )


def test_general_path_emits_two_requantizes():
    """
    lhs_scale != rhs_scale -- fast path must NOT fire, and the canonicalized
    graph should contain the original TWO qnn.requantize nodes (one per
    operand), proving the general/fallback path is untouched.
    """
    body = canonicalize_qnn_subtract(
        lhs_scale=0.2, lhs_zp=0, rhs_scale=0.3, rhs_zp=0, out_scale=0.5, out_zp=5
    )
    counts = count_ops(body)

    assert counts.get("fixed_point_multiply", 0) == 2, (
        f"Expected exactly 2 fixed_point_multiply ops (requantize proxy) when "
        f"lhs/rhs have differing qnn params (general path), got "
        f"{counts.get('fixed_point_multiply', 0)}. Full op counts: {counts}."
    )


def test_fast_path_zero_points_also_equal_required():
    """
    lhs_scale == rhs_scale but lhs_zero_point != rhs_zero_point -- the fast
    path guard requires BOTH scale and zero_point to match, so this should
    still take the general (2-requantize) path.
    """
    body = canonicalize_qnn_subtract(
        lhs_scale=0.2, lhs_zp=0, rhs_scale=0.2, rhs_zp=5, out_scale=0.5, out_zp=5
    )
    counts = count_ops(body)

    assert counts.get("fixed_point_multiply", 0) == 2, (
        f"Expected general path (2 fixed_point_multiply ops) when only scale "
        f"matches but zero_point differs, got {counts.get('fixed_point_multiply', 0)}. "
        f"Full op counts: {counts}."
    )


def test_fast_path_reduces_total_op_count():
    """
    Sanity check on total graph size: the fast path's graph (cast, cast,
    subtract, requantize, cast) should have strictly fewer total call nodes
    than the general path's graph (requantize, requantize, subtract, add,
    cast) for an equivalent scenario, confirming the optimization actually
    shrinks the graph rather than just reshuffling op types.
    """
    fast_body = canonicalize_qnn_subtract(
        lhs_scale=0.2, lhs_zp=0, rhs_scale=0.2, rhs_zp=0, out_scale=0.5, out_zp=5
    )
    general_body = canonicalize_qnn_subtract(
        lhs_scale=0.2, lhs_zp=0, rhs_scale=0.3, rhs_zp=0, out_scale=0.5, out_zp=5
    )

    fast_total = sum(count_ops(fast_body).values())
    general_total = sum(count_ops(general_body).values())

    assert fast_total < general_total, (
        f"Expected fast path to produce a smaller graph than the general path "
        f"(fewer total ops), got fast={fast_total} vs general={general_total}."
    )


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
    