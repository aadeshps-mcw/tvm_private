#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class PowSimplifier : public StmtExprMutator {
 public:
  PrimExpr VisitExpr_(const CallNode* op) final {
    auto expr = StmtExprMutator::VisitExpr_(op);

    const CallNode* call = expr.as<CallNode>();
    if (call && call->op == tvm::Op::Get("tir.pow")) {
      PrimExpr base = call->args[0];
      PrimExpr exp = call->args[1];

     double val = 0.0;
      bool is_const_exponent = false;

      // 2. Type-Agnostic Extraction
      if (const IntImmNode* int_imm = exp.as<IntImmNode>()) {
        val = static_cast<double>(int_imm->value);
        is_const_exponent = true;
      } else if (const FloatImmNode* float_imm = exp.as<FloatImmNode>()) {
        val = float_imm->value;
        is_const_exponent = true;
      }

      // 3. Apply optimizations if we successfully extracted a constant
      if (is_const_exponent) {

        if(val == 0.0f) {
          return make_const(base.dtype(), 1.0f);
        }

        if(val == 1.0f) {
          return base;
        }

        if (val == 2.0f) {
          return Mul(base, base);
        }

        if(val == 0.5f) {
          return sqrt(base);
        }

        if (val == -0.5f) {
          return make_const(base.dtype(), 1.0f) / sqrt(base);
        }

        if(val == 3.0f) {
          return Mul(Mul(base, base), base);
        }

        if(val == -1.0f) {
          return make_const(base.dtype(), 1.0f) / base;
        }
      }
    }

    return expr;
  }
};

namespace transform {

Pass SimplifyPow() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = PowSimplifier()(std::move(n->body));
    return f;
  };

  return CreatePrimFuncPass(pass_func, 0, "tir.SimplifyPow", {});
}


// Register the pass
TVM_REGISTER_GLOBAL("tir.transform.SimplifyPow")
    .set_body_typed(SimplifyPow);

}  // namespace transform

}  // namespace tir
}  // namespace tvm