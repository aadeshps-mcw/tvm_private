# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Consist of utilities and methods for lowering QNN into mainline relay."""
from typing import Callable

import numpy as np
import tvm
from tvm import relay


def run_const_expr(expr: "relay.Expr") -> np.ndarray:
    """Evaluate a const expression, receiving result as np array.

    If a number of passes are disabled in the current Pass Context, then there is no need to disable
    these passes for const expression evaluation as well. That's why we use empty list
    "disabled_pass=[]", all other arguments are inherited from the current Pass Context.
    """
    curr_pass_ctx = tvm.ir.transform.PassContext.current()
    with tvm.ir.transform.PassContext(
        opt_level=curr_pass_ctx.opt_level,
        required_pass=curr_pass_ctx.required_pass,
        disabled_pass=[],
        instruments=curr_pass_ctx.instruments,
        config=curr_pass_ctx.config,
    ):
        mod = tvm.IRModule.from_expr(expr)
        vm_exe = relay.create_executor("vm", mod=mod)
        output = vm_exe.evaluate()().asnumpy()

    return output


def create_integer_lookup_table(
    floating_point_func: Callable[[np.ndarray], np.ndarray],
    input_scale: "relay.Expr",
    input_zero_point: "relay.Expr",
    output_scale: "relay.Expr",
    output_zero_point: "relay.Expr",
    in_axis: int = -1,
    out_axis: int = -1,
    in_dtype: str = "uint8",
    out_dtype: str = "uint8",
) -> np.ndarray:
    """
    Return a table where each input indexes to the output quantizing the given function.

    Note this also supports mapping unsigned and signed integers to each other.

    Args:
      floating_point_func: The numpy function which this table is to approximate
      input_scale: The scale of the quantized input tensor.
      input_zero_point: The zero point of the quantized input tensor.
      output_scale: The scale of the quantized output tensor.
      output_zero_point: The zero point of the quantized output tensor.
      in_axis: The axis for multi-channel quantization of the input if applicable.
      out_axis: The axis for multi-channel quantization of the output if applicable.
      in_dtype: The dtype of the input tensor.
      out_dtype: The wanted dtype of the output tensor.

    Returns:
      A numpy array where values in quantized space will index to the output in quantized space
      approximating the given function.
    """
    if not np.issubdtype(np.dtype(in_dtype), np.integer) or not np.issubdtype(
        np.dtype(out_dtype), np.integer
    ):
        raise ValueError("Only integer dtypes allowed.")

    num_bits = np.iinfo(in_dtype).bits
    in_scale_np = input_scale.data.numpy()
    
    # Detect if quantization is per-channel
    is_per_channel = len(in_scale_np.shape) > 0 and in_scale_np.shape[0] > 1

    # Build 1D (256) or 2D (C, 256) base array
    base_quantized = np.array(range(0, 2**num_bits), dtype=f"uint{num_bits}")
    if is_per_channel:
        num_channels = in_scale_np.shape[0]
        inputs_quantized = np.tile(base_quantized, (num_channels, 1))
        lut_axis = 0 # The channel dimension in our new 2D LUT
    else:
        inputs_quantized = base_quantized
        lut_axis = in_axis

    inputs_quantized = inputs_quantized.view(in_dtype)
    inputs_quantized = relay.const(inputs_quantized, dtype=in_dtype)
    
    # Process the table through the float function
    inputs_dequantized = run_const_expr(
        relay.qnn.op.dequantize(
            inputs_quantized,
            input_scale=input_scale,
            input_zero_point=input_zero_point,
            axis=lut_axis,
        )
    )

    output_dequantized = relay.const(floating_point_func(inputs_dequantized))
    
    output_quantized = run_const_expr(
        relay.qnn.op.quantize(
            output_dequantized, 
            output_scale, 
            output_zero_point, 
            axis=lut_axis if is_per_channel else out_axis, 
            out_dtype=out_dtype
        )
    )

    return output_quantized


def create_integer_lookup_op(
    input_arg: "relay.Expr",
    floating_point_func: Callable[[np.array], np.array],
    in_scale: "relay.Expr",
    in_zero_point: "relay.Expr",
    out_scale: "relay.Expr",
    out_zero_point: "relay.Expr",
    in_axis: int = -1,
    out_axis: int = -1,
    in_dtype: str = "uint8",
    out_dtype: str = "uint8",
) -> "relay.Expr":
    """Create a quantized version of the given floating point unary operation using table lookup."""
    
    in_scale_np = in_scale.data.numpy()
    is_per_channel = len(in_scale_np.shape) > 0 and in_scale_np.shape[0] > 1

    # Safely extract arrays without crashing on .item()
    if is_per_channel:
        in_s_val, in_zp_val = in_scale_np, in_zero_point.data.numpy()
        out_s_val, out_zp_val = out_scale.data.numpy(), out_zero_point.data.numpy()
    else:
        in_s_val, in_zp_val = in_scale_np.item(), in_zero_point.data.numpy().item()
        out_s_val, out_zp_val = out_scale.data.numpy().item(), out_zero_point.data.numpy().item()

    # Generate the LUT (will be 2D if per-channel)
    lookup_table = create_integer_lookup_table(
        floating_point_func,
        relay.const(in_s_val),
        relay.const(in_zp_val, dtype="int32"),
        relay.const(out_s_val),
        relay.const(out_zp_val, dtype="int32"),
        in_axis=in_axis, out_axis=out_axis,
        in_dtype=in_dtype, out_dtype=out_dtype,
    )

    in_dtype_num_bits = np.iinfo(in_dtype).bits
    index_tensor = relay.reinterpret(input_arg, f"uint{in_dtype_num_bits}")

    # Apply the LUT
    if not is_per_channel:
        return relay.take(relay.const(lookup_table), index_tensor, axis=0, mode="fast")
    else:
        # Dynamically derive shape and rank from the Relay Expression
        in_shape = None
        try:
            if input_arg.checked_type is not None:
                in_shape = input_arg.checked_type.shape
        except ValueError:
            # Type checker hasn't run yet; safe to proceed to fallback
            pass

        if in_shape is None:
            if hasattr(input_arg, "type_annotation") and input_arg.type_annotation is not None:
                in_shape = input_arg.type_annotation.shape
            else:
                raise TypeError(
                    "input_arg must have an inferred type or type annotation to resolve per-channel lookup shapes."
                )
            
        # Convert TVM IntImm/tir values to python integers for calculation
        in_shape = [int(dim) for dim in in_shape]
        rank = len(in_shape)
        # --------------------------------------------------------------------------

        C = in_scale_np.shape[0]
        
        # Create offsets: [0, 256, 512, ...]
        offsets = np.arange(C, dtype="int32") * (2 ** in_dtype_num_bits)
        
        # Reshape offsets to broadcast correctly across the input tensor (e.g. 1xCx1x1)
        actual_axis = in_axis if in_axis >= 0 else rank + in_axis
        offset_shape = [1] * rank
        offset_shape[actual_axis] = C
        offsets = offsets.reshape(offset_shape)
        
        # Add offsets to the tensor and do a flat 1D take
        index_tensor_int32 = relay.cast(index_tensor, "int32")
        shifted_indices = relay.add(index_tensor_int32, relay.const(offsets, dtype="int32"))
        
        flat_lookup_table = relay.const(lookup_table.flatten())
        return relay.take(flat_lookup_table, shifted_indices, axis=0, mode="fast")
