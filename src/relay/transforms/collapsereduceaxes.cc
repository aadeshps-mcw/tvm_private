/*
 * If a reduce op has effective axis == [0],
 * collapse all other dimensions into one,
 * perform reduction, then reshape back
 * to the correct output shape.
 */

#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include "../op/tensor/transform.h"
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/expr_functor.h>

#include <vector>
#include <algorithm>

namespace tvm {
namespace relay {

// Forward declaration (from reduce.cc)
std::vector<int64_t> GetReduceAxes(uint32_t indim,
                                   const Array<Integer>& axis,
                                   bool exclude);


bool IsTargetReduceOp(const Op& op) {
  static auto sum_op  = Op::Get("sum");
  static auto max_op  = Op::Get("max");
  static auto min_op  = Op::Get("min");
  static auto mean_op = Op::Get("mean");
  static auto all_op  = Op::Get("all");
  static auto any_op  = Op::Get("any");

  return op == sum_op || op == max_op ||
         op == min_op || op == mean_op ||
         op == all_op || op == any_op;
}

bool IsOnlyAxis0(const std::vector<int64_t>& axes) {
  return axes.size() == 1 && axes[0] == 0;
}

Array<PrimExpr> CollapseShape(const Array<PrimExpr>& shape) {
  ICHECK_GE(shape.size(), 2);
  PrimExpr dim0 = shape[0];
  PrimExpr rest = shape[1];
  for (size_t i = 2; i < shape.size(); ++i) {
    rest = rest * shape[i];
  }
  return {dim0, rest};
}


Array<PrimExpr> ComputeOutputShape(const Array<PrimExpr>& in_shape,
                                   bool keepdims) {
  Array<PrimExpr> out;
  if (keepdims) {
    out.push_back(Integer(1));
    for (size_t i = 1; i < in_shape.size(); ++i) {
      out.push_back(in_shape[i]);
    }
  } else {
    for (size_t i = 1; i < in_shape.size(); ++i) {
      out.push_back(in_shape[i]);
    }
  }
  return out;
}


Array<Integer> ConvertToIntegerArray(const Array<PrimExpr>& shape) {
  Array<Integer> res;
  for (auto s : shape) {
    if (auto imm = s.as<IntImmNode>()) {
      res.push_back(Integer(static_cast<int>(imm->value)));
    } else {
      LOG(FATAL) << "Dynamic shape not supported in this pass";
    }
  }
  return res;
}

class CollapseReduceMutator : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* call) override {
    Expr new_expr = ExprMutator::VisitExpr_(call);
    call = new_expr.as<CallNode>();
    auto op = call->op.as<OpNode>();
    if (!op || !IsTargetReduceOp(GetRef<Op>(op))) {
      return new_expr;
    }

    auto attrs = call->attrs.as<ReduceAttrs>();
    auto input_ty = call->args[0]->checked_type().as<TensorTypeNode>();
    uint32_t indim = input_ty->shape.size();
    auto axes = GetReduceAxes(indim, attrs->axis, attrs->exclude);

    if (!IsOnlyAxis0(axes)) {
      return new_expr;
    }
    if (indim <= 1) {
      return new_expr;
    }
    auto orig_shape = input_ty->shape;
    auto collapsed_shape = CollapseShape(orig_shape);
    Expr reshaped = MakeReshape(call->args[0],  
            ConvertToIntegerArray(collapsed_shape), false);
    auto new_attrs = make_object<ReduceAttrs>(*attrs);
    new_attrs->axis = Array<Integer>({0});
    new_attrs->exclude = false;

    Expr reduced = Call(call->op, {reshaped}, Attrs(new_attrs), {});
    auto final_shape = ComputeOutputShape(orig_shape, attrs->keepdims);
    return MakeReshape(reduced, ConvertToIntegerArray(final_shape), false);
  }
};

namespace transform {

Pass CollapseReduceAxis0Pass() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [](Function f, IRModule m, PassContext pc) -> Function {
        CollapseReduceMutator mutator;
        return Downcast<Function>(mutator.VisitExpr(f));
      };

  return CreateFunctionPass(pass_func,
                            0,
                            "CollapseReduceAxis0Pass",
                            {"InferType"});
}


TVM_REGISTER_GLOBAL("relay._transform.CollapseReduceAxis0Pass")
    .set_body_typed(CollapseReduceAxis0Pass);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
