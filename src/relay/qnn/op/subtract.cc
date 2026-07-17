/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/qnn/op/subtract.cc
 * \brief QNN subtract operator.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>

#include "op_common.h"

namespace tvm {
namespace relay {
namespace qnn {

/*
 * \brief Canonicalizes the QNN subtract op.
 * \param attrs The empty attribute.
 * \param new_args The new mutated args to the call node.
 * \param arg_types The types of input and output.
 * \return The sequence of Relay ops for add op.
 */
Expr QnnSubtractCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                             const Array<tvm::relay::Type>& arg_types) {
  // Get the args.
  QnnBinaryOpArguments args(new_args);

  // Get the input dtype and shape.
  QnnBinaryOpTensorType input_type(arg_types, 0);

  const auto* broadcast_attrs = attrs.as<BroadcastAttrs>();
  ICHECK(broadcast_attrs != nullptr);

  auto lhs_axis = broadcast_attrs->lhs_axis;
  auto rhs_axis = broadcast_attrs->rhs_axis;

  // Fast path: if lhs and rhs already share identical qnn params (scale and
  // zero_point), the zero points cancel out algebraically:
  //
  //   scale * (Q_a - zp) - scale * (Q_b - zp) = scale * (Q_a - Q_b)
  //
  // so we can subtract the raw quantized values directly (upcast to int32 to
  // avoid overflow) and only requantize ONCE at the end, from lhs's scale
  // into the output's qnn params -- instead of requantizing both operands
  // up front into the output's params before subtracting. This avoids one
  // full requantize op and avoids the extra rounding error introduced by
  // rounding both operands independently before subtracting.
  //
  // IsEqualScalar only recognizes literal scalar constants, so this fast
  // path naturally only fires for per-tensor quantization where both sides'
  // scale/zero_point are equal compile-time constants. Per-channel
  // quantization (tensor-valued scales) or differing params safely fall
  // through to the general path below.
  if (IsEqualScalar(args.lhs_scale, args.rhs_scale) &&
      IsEqualScalar(args.lhs_zero_point, args.rhs_zero_point)) {
    // Upcast both operands to int32 before subtracting, to avoid overflow
    // (e.g. uint8 - uint8 can be negative, which doesn't fit back in uint8).
    auto lhs_int32 = Cast(args.lhs, DataType::Int(32));
    auto rhs_int32 = Cast(args.rhs, DataType::Int(32));

    // Computes Q_a - Q_b directly; the shared zero_point has already
    // cancelled out algebraically, so this result is expressed in
    // (lhs_scale, zero_point=0).
    auto diff = Subtract(lhs_int32, rhs_int32);

    // Single requantize: from (lhs_scale, 0) to the output's qnn params.
    auto zero_zp = MakeConstantScalar(DataType::Int(32), 0);
    auto requantized_diff =
        RequantizeOrUpcast(diff, args.lhs_scale, zero_zp, args.output_scale,
                           args.output_zero_point, input_type.shape, lhs_axis);

    return ConvertDtype(requantized_diff, input_type.dtype);
  }

  // General path (lhs/rhs qnn params differ):
  //
  // Since the input qnn params can be different than output qnn params, we first requantize the
  // input tensors to the output qnn params. Then we call relay.subtract on the requantized inputs.
  // This subtraction results in extra subtraction of the output zero point. We further add
  // the zero point. The whole process can be represented using following equations
  //
  //          scale_c * (Q_c - zp_c) = scale_a * (Q_a - zp_a) - scale_b * (Q_b - zp_b)
  //
  // After requantizing Q_a and Q_b, equation becomes,
  //          scale_c * (Q_c - zp_c) = scale_c * (Q_a' - zp_c) - scale_c * (Q_b' - zp_c)
  //          scale_c * (Q_c - zp_c) = scale_c * (Q_a' - Q_b')
  //
  // Comparing the LHS and RHS, it results in
  //          Q_c = Q_a' - Q_b' + zp_c
  // The subtract op is done in int32 precision.

  // Requantize LHS if necessary. Computes Q_a'
  auto requantized_lhs =
      RequantizeOrUpcast(args.lhs, args.lhs_scale, args.lhs_zero_point, args.output_scale,
                         args.output_zero_point, input_type.shape, lhs_axis);
  // Requantize RHS if necessary. Computes Q_b'
  auto requantized_rhs =
      RequantizeOrUpcast(args.rhs, args.rhs_scale, args.rhs_zero_point, args.output_scale,
                         args.output_zero_point, input_type.shape, rhs_axis);

  // Computes Q_a' - Q_b'
  auto output = Subtract(requantized_lhs, requantized_rhs);

  // Add zero point. Computes (Q_a' - Q_b') + zp_c
  auto zero_scalar = MakeConstantScalar(DataType::Int(32), 0);
  if (!IsEqualScalar(args.output_zero_point, zero_scalar)) {
    output = Add(output, args.output_zero_point);
  }

  // Go back to lower precision.
  return ConvertDtype(output, input_type.dtype);
}

// QNN Subtraction operator.
QNN_REGISTER_BINARY_OP("subtract")
    .describe("Elementwise subtract with broadcasting for quantized tensors.")
    .set_support_level(11)
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnSubtractCanonicalize)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm