#include <tvm/relay/attrs/algorithm.h>
#include <tvm/relay/op.h>
#include <tvm/relay/type.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/registry.h>

#include <tvm/te/operation.h>
#include <tvm/topi/nn.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(BlackmanWindowAttrs);

bool BlackmanWindowRel(const Array<Type>& types,
                       int num_inputs,
                       const Attrs& attrs,
                       const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 1);

  const auto* param = attrs.as<BlackmanWindowAttrs>();
  ICHECK(param != nullptr);

  ICHECK_GT(param->window_length, 0)
      << "Blackman window length must be > 0";

  ICHECK(param->dtype.is_float())
      << "Blackman window only supports floating-point dtypes.";

  Array<IndexExpr> oshape = {
      tir::make_const(DataType::Int(64), param->window_length)
  };

  reporter->Assign(types[0], TensorType(oshape, param->dtype));
  return true;
}


RELAY_REGISTER_OP("blackman_window")
    .describe("Generate a Blackman window.")
    .set_num_inputs(0)
    .set_attrs_type<BlackmanWindowAttrs>()
    .add_type_rel("BlackmanWindowRel", BlackmanWindowRel)
    .set_attr<TOpPattern>("TOpPattern", kElemWise)
    .set_support_level(6);

Expr MakeBlackmanWindow(int window_length,
                        Bool periodic,
                        DataType dtype) {
  auto attrs = make_object<BlackmanWindowAttrs>();
  attrs->window_length = window_length;
  attrs->periodic = periodic;
  attrs->dtype = dtype;

  static const Op& op = Op::Get("blackman_window");
  return Call(op, {}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make.blackman_window")
    .set_body_typed(MakeBlackmanWindow);

}  // namespace relay
}  // namespace tvm