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
"""Generic image operators"""
from .default import default_schedule as _default_schedule
from tvm import te
from tvm.topi.utils import get_const_int

def _apply_adaptive_spatial_schedule(s, Out, outer_axes, tile_axis, tile_dim_length, red_axes, vec_axis, vec_dim_length):
    """
    Unified scheduling logic that applies dynamic vector alignment 
    and spatial strip-mining for cache locality.
    """
    if tile_dim_length <= 16:
            return s

    # Dynamic Vector Width Alignment
    vec_width = 1
    if vec_dim_length >= 16:
        if vec_dim_length % 8 == 0:
            vec_width = 8
        elif vec_dim_length % 4 == 0:
            vec_width = 4

    vectorizable = vec_width > 1
    if vectorizable:
        vo, vi = s[Out].split(vec_axis, factor=vec_width)

    # Dynamic Strip-Mining (Cache Tiling)
    if tile_dim_length >= 128:
        tile_factor = 32 if tile_dim_length % 32 == 0 else 16
    elif tile_dim_length >= 64:
        tile_factor = 16
    else:
        tile_factor = 1

    #Apply Splits, Reorders, and Fusions
    if tile_factor > 1:
        to, ti = s[Out].split(tile_axis, factor=tile_factor)
        
        if vectorizable:
            # Reorder: [Outer Axes] -> Outer Tile -> [Kernel] -> Inner Tile -> Outer Vec -> Inner Vec
            s[Out].reorder(*(outer_axes + [to] + red_axes + [ti, vo, vi]))
            s[Out].vectorize(vi)
        else:
            s[Out].reorder(*(outer_axes + [to] + red_axes + [ti, vec_axis]))
            
        fused = s[Out].fuse(*(outer_axes + [to]))
    else:
        # Fallback for short dimensions
        if vectorizable:
            s[Out].reorder(*(outer_axes + [tile_axis] + red_axes + [vo, vi]))
            s[Out].vectorize(vi)
        else:
            s[Out].reorder(*(outer_axes + [tile_axis] + red_axes + [vec_axis]))
            
        fused = s[Out].fuse(*outer_axes)

    # Unroll Reduction Axes
    for r_axis in red_axes:
        s[Out].unroll(r_axis)

    # 5. Parallelize
    s[Out].parallel(fused)
    
    return s

def schedule_dilation2d_nchw(outs):
    """Adaptive schedule for image.dilation2d, NCHW data layout."""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = _default_schedule(outs, False)
    Out = outs[0]

    n, c, h, w = s[Out].op.axis
    ry, rx = s[Out].op.reduce_axis

    try:
         out_h = get_const_int(Out.shape[2])
         out_w = get_const_int(Out.shape[3])
    except ValueError:
         return s

    # NCHW Role Mapping:
    # We fuse Batch & Channels (n, c)
    # We tile Height (h)
    # We vectorize Width (w)
    return _apply_adaptive_spatial_schedule(
        s=s, Out=Out,
        outer_axes=[n, c],
        tile_axis=h, tile_dim_length=out_h,
        red_axes=[ry, rx],
        vec_axis=w, vec_dim_length=out_w
    )


def schedule_dilation2d_nhwc(outs):
    """Adaptive schedule for image.dilation2d, NHWC data layout."""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = _default_schedule(outs, False)
    Out = outs[0]

    n, h, w, c = s[Out].op.axis
    ry, rx = s[Out].op.reduce_axis

    try:
         out_w = get_const_int(Out.shape[2])
         channels = get_const_int(Out.shape[3])
    except ValueError:
         return s

    # NHWC Role Mapping:
    # We fuse Batch & Height (n, h)
    # We tile Width (w) for kernel reuse
    # We vectorize Channels (c)
    return _apply_adaptive_spatial_schedule(
        s=s, Out=Out,
        outer_axes=[n, h],
        tile_axis=w, tile_dim_length=out_w,
        red_axes=[ry, rx],
        vec_axis=c, vec_dim_length=channels
    )
# def schedule_dilation2d_nchw(outs):
#     """Schedule for dilation2d
#     Parameters
#     ----------
#     outs : Array of Tensor
#         The computation graph description of dilation2d
#         in the format of an array of tensors.
#     Returns
#     -------
#     sch : Schedule
#         The computation schedule for the op.
#     """
#     return _default_schedule(outs, False)

# def schedule_dilation2d_nhwc(outs):
#     """Schedule for dilation2d
#     Parameters
#     ----------
#     outs : Array of Tensor
#         The computation graph description of dilation2d
#         in the format of an array of tensors.
#     Returns
#     -------
#     sch : Schedule
#         The computation schedule for the op.
#     """
#     return _default_schedule(outs, False)