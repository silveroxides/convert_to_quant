import torch
import math
from typing import List, Tuple, Union, Optional, Dict, Any
from .constants import dtype_dict, linear_types, conv_types, conv_transpose_types

def get_scale_asymmetric(weight: torch.FloatTensor, reduction_axes: Union[int, List[int]], weights_dtype: str) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    zero_point = torch.amin(weight, dim=reduction_axes, keepdims=True)
    scale = torch.amax(weight, dim=reduction_axes, keepdims=True).sub_(zero_point).div_(dtype_dict[weights_dtype]["max"] - dtype_dict[weights_dtype]["min"])
    if dtype_dict[weights_dtype]["min"] != 0:
        zero_point.sub_(torch.mul(scale, dtype_dict[weights_dtype]["min"]))
    return scale, zero_point

def get_scale_symmetric(weight: torch.FloatTensor, reduction_axes: Union[int, List[int]], weights_dtype: str) -> torch.FloatTensor:
    return torch.amax(weight.abs(), dim=reduction_axes, keepdims=True).div_(dtype_dict[weights_dtype]["max"])

def quantize_weight(weight: torch.FloatTensor, reduction_axes: Union[int, List[int]], weights_dtype: str, use_stochastic_rounding: bool = False) -> Tuple[torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
    weight = weight.to(dtype=torch.float32)

    if dtype_dict[weights_dtype]["is_unsigned"]:
        scale, zero_point = get_scale_asymmetric(weight, reduction_axes, weights_dtype)
        quantized_weight = torch.sub(weight, zero_point).div_(scale)
    else:
        scale = get_scale_symmetric(weight, reduction_axes, weights_dtype)
        quantized_weight = torch.div(weight, scale)
        zero_point = None

    if dtype_dict[weights_dtype]["is_integer"]:
        if use_stochastic_rounding:
            quantized_weight.add_(torch.rand_like(quantized_weight), alpha=0.1)
        quantized_weight.round_()
    else:
        if use_stochastic_rounding:
            mantissa_difference = 1 << (23 - dtype_dict[weights_dtype]["mantissa"])
            quantized_weight = quantized_weight.view(dtype=torch.int32).add_(torch.randint_like(quantized_weight, low=0, high=mantissa_difference, dtype=torch.int32)).view(dtype=torch.float32)
        quantized_weight.nan_to_num_()
    
    quantized_weight = quantized_weight.clamp_(dtype_dict[weights_dtype]["min"], dtype_dict[weights_dtype]["max"])
    return quantized_weight, scale, zero_point

def apply_svdquant(weight: torch.FloatTensor, rank: int = 32, niter: int = 8) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    reshape_weight = False
    if weight.ndim > 2: # convs
        reshape_weight = True
        weight_shape = weight.shape
        weight = weight.flatten(1,-1)
    weight = weight.to(dtype=torch.float32)
    U, S, svd_down = torch.svd_lowrank(weight, q=rank, niter=niter)
    svd_up = torch.mul(U, S.unsqueeze(0))
    svd_down = svd_down.t_()
    weight = weight.sub(torch.mm(svd_up, svd_down))
    if reshape_weight:
        weight = weight.unflatten(-1, (*weight_shape[1:],))
    return weight, svd_up, svd_down

def prepare_weight_for_matmul(weight: torch.Tensor, use_contiguous_mm: bool = False) -> torch.Tensor:
    if use_contiguous_mm:
        weight = weight.contiguous()
    elif weight.is_contiguous():
        weight = weight.t_().contiguous().t_()
    return weight

def prepare_svd_for_matmul(svd_up: torch.FloatTensor, svd_down: torch.FloatTensor, use_quantized_matmul: bool, use_contiguous_mm: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    if svd_up is not None:
        if use_quantized_matmul:
            svd_up = prepare_weight_for_matmul(svd_up, use_contiguous_mm)
        else:
            svd_up = svd_up.contiguous()
    if svd_down is not None:
        svd_down = prepare_weight_for_matmul(svd_down, use_contiguous_mm)
    return svd_up, svd_down

# Integer packing functions
def pack_uint7(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.bitwise_or(
        packed_tensor[:, :7],
        torch.bitwise_and(
            torch.stack(
                (
                    torch.bitwise_left_shift(packed_tensor[:, 7], 1),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 2),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 3),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 4),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 5),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 6),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 7),
                ),
                dim=-1
            ),
            128
        ),
    )
    return packed_tensor

def pack_uint6(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 4)
    packed_tensor = torch.cat(
        (
            torch.bitwise_or(
                packed_tensor[:, :2],
                torch.bitwise_and(
                    torch.stack(
                        (
                            torch.bitwise_left_shift(packed_tensor[:, 3], 2),
                            torch.bitwise_left_shift(packed_tensor[:, 3], 4),
                        ),
                        dim=-1
                    ),
                    192
                )
            ),
            torch.bitwise_or(packed_tensor[:, 2], torch.bitwise_left_shift(packed_tensor[:, 3], 6)).unsqueeze(-1),
        ),
        dim=-1
    )
    return packed_tensor

def pack_uint5(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.cat(
        (
            torch.bitwise_or(packed_tensor[:, :3], torch.bitwise_left_shift(packed_tensor[:, 5:8], 5)),
            torch.bitwise_or(
                packed_tensor[:, 3],
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 5], 2), 96),
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 7], 3), 128),
                ),
            ).unsqueeze(-1),
            torch.bitwise_or(
                packed_tensor[:, 4],
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 6], 2), 96),
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 7], 4), 128),
                ),
            ).unsqueeze(-1),
        ),
        dim=-1
    )
    return packed_tensor

def pack_uint4(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 2)
    packed_tensor = torch.bitwise_or(packed_tensor[:, 0], torch.bitwise_left_shift(packed_tensor[:, 1], 4))
    return packed_tensor

def pack_uint3(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.bitwise_or(
        torch.bitwise_or(packed_tensor[:, :3], torch.bitwise_left_shift(packed_tensor[:, 3:6], 3)),
        torch.cat(
            (
                torch.bitwise_left_shift(packed_tensor[:, 6:8], 6),
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 6], 4), 64),
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 7], 5), 128),
                ).unsqueeze(-1),
            ),
            dim=-1
        )
    )
    return packed_tensor

def pack_uint2(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 4)
    packed_tensor = torch.bitwise_or(
        torch.bitwise_or(packed_tensor[:, 0], torch.bitwise_left_shift(packed_tensor[:, 1], 2)),
        torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 2], 4), torch.bitwise_left_shift(packed_tensor[:, 3], 6)),
    )
    return packed_tensor

def pack_uint1(tensor: torch.Tensor) -> torch.Tensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.bitwise_or(
        torch.bitwise_or(
            torch.bitwise_or(packed_tensor[:, 0], torch.bitwise_left_shift(packed_tensor[:, 1], 1)),
            torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 2], 2), torch.bitwise_left_shift(packed_tensor[:, 3], 3))
        ),
        torch.bitwise_or(
            torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 4], 4), torch.bitwise_left_shift(packed_tensor[:, 5], 5)),
            torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 6], 6), torch.bitwise_left_shift(packed_tensor[:, 7], 7))
        ),
    )
    return packed_tensor

packed_int_function_dict = {
    "int7": pack_uint7, "int6": pack_uint6, "int5": pack_uint5, "int4": pack_uint4, "int3": pack_uint3, "int2": pack_uint2,
    "uint7": pack_uint7, "uint6": pack_uint6, "uint5": pack_uint5, "uint4": pack_uint4, "uint3": pack_uint3, "uint2": pack_uint2,
    "uint1": pack_uint1, "bool": pack_uint1,
}

def pack_int_symetric(tensor: torch.Tensor, weights_dtype: str) -> torch.Tensor:
    return packed_int_function_dict[weights_dtype](tensor.sub_(dtype_dict[weights_dtype]["min"]).to(dtype=dtype_dict[weights_dtype]["storage_dtype"]))

def pack_int_asymetric(tensor: torch.Tensor, weights_dtype: str) -> torch.Tensor:
    return packed_int_function_dict[weights_dtype](tensor.to(dtype=dtype_dict[weights_dtype]["storage_dtype"]))

# Float packing logic
float_bits_to_uint_dict = {1: "uint1", 2: "uint2", 3: "uint3", 4: "uint4", 5: "uint5", 6: "uint6", 7: "uint7"}

def pack_float(x: torch.FloatTensor, weights_dtype: str) -> torch.Tensor:
    exponent_bits = dtype_dict[weights_dtype]["exponent"]
    mantissa_bits = dtype_dict[weights_dtype]["mantissa"]
    total_bits = dtype_dict[weights_dtype]["num_bits"]

    if dtype_dict[weights_dtype]["is_unsigned"]:
        sign_mask = (1 << (total_bits-1))
    else:
        sign_mask = (1 << (total_bits-1)) + (1 << (total_bits-2))

    mantissa_difference = 23 - mantissa_bits
    exponent_difference = 8 - exponent_bits
    mantissa_mask = (1 << mantissa_difference)

    x = x.to(dtype=torch.float32).view(torch.int32)
    x = torch.where(
        torch.gt(
            torch.bitwise_and(x, -(1 << (mantissa_difference-4)) & ~(-mantissa_mask)),
            (1 << (mantissa_difference-1)),
        ),
        torch.add(x, mantissa_mask),
        x,
    )
    x = torch.where(torch.lt(x.view(torch.float32).abs(), dtype_dict[weights_dtype].get("min_normal", 0)), 0, x)
    x = torch.bitwise_right_shift(x, mantissa_difference)
    x = torch.bitwise_and(
        torch.bitwise_or(
            torch.bitwise_and(torch.bitwise_right_shift(x, exponent_difference), sign_mask),
            torch.bitwise_and(x, ~sign_mask),
        ),
        ~(-(1 << total_bits)),
    ).view(torch.uint32)

    if total_bits < 8:
        x = pack_int_asymetric(x, float_bits_to_uint_dict[total_bits])
    else:
        x = x.to(dtype=dtype_dict[weights_dtype]["storage_dtype"])
    return x

def sdnq_quantize_layer_weight(
    weight: torch.Tensor,
    layer_class_name: str = None,
    settings: dict = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
    """
    Stateless implementation of SDNQ layer quantization.
    Returns (quantized_weight, scale, zero_point, svd_up, svd_down, info)
    """
    if settings is None:
        settings = {}
    
    weights_dtype = settings.get("weights_dtype", "int8")
    quantized_matmul_dtype = settings.get("quantized_matmul_dtype", None)
    torch_dtype = settings.get("torch_dtype", weight.dtype)
    group_size = settings.get("group_size", 0)
    svd_rank = settings.get("svd_rank", 32)
    svd_steps = settings.get("svd_steps", 8)
    use_svd = settings.get("use_svd", False)
    use_quantized_matmul = settings.get("use_quantized_matmul", False)
    use_stochastic_rounding = settings.get("use_stochastic_rounding", False)
    dequantize_fp32 = settings.get("dequantize_fp32", False)
    using_pre_calculated_svd = settings.get("using_pre_calculated_svd", False)
    use_tensorwise_fp8_matmul = settings.get("use_tensorwise_fp8_matmul", True)
    use_contiguous_mm = settings.get("use_contiguous_mm", False)

    num_of_groups = 1
    is_conv_type = False
    is_conv_transpose_type = False
    is_linear_type = False
    result_shape = None
    original_shape = weight.shape
    weight = weight.detach()

    if quantized_matmul_dtype is None:
        if dtype_dict[weights_dtype]["is_integer"]:
            quantized_matmul_dtype = "int8"
        elif dtype_dict[weights_dtype]["num_bits"] == 8:
            quantized_matmul_dtype = "float8_e4m3fn"
        else:
            quantized_matmul_dtype = "float16"

    re_quantize_for_matmul = bool(
        dtype_dict[weights_dtype]["is_unsigned"]
        or dtype_dict[weights_dtype]["is_integer"] != dtype_dict[quantized_matmul_dtype]["is_integer"]
        or dtype_dict[weights_dtype]["num_bits"] > dtype_dict[quantized_matmul_dtype]["num_bits"]
        or (
            dtype_dict[weights_dtype]["is_packed"]
            and not dtype_dict[weights_dtype]["is_integer"]
            and not dtype_dict[quantized_matmul_dtype]["is_integer"]
            and (
                    dtype_dict[weights_dtype]["num_bits"] >= dtype_dict[quantized_matmul_dtype]["num_bits"]
                    or dtype_dict[weights_dtype]["max"] > dtype_dict[quantized_matmul_dtype]["max"]
                )
        )
    )

    if layer_class_name in conv_types:
        is_conv_type = True
        reduction_axes = 1
        output_channel_size, channel_size = weight.shape[:2]
        if use_quantized_matmul:
            use_quantized_matmul = channel_size >= 32 and output_channel_size >= 32
            use_quantized_matmul = use_quantized_matmul and output_channel_size % 16 == 0 and channel_size % 16 == 0
        if use_quantized_matmul and not re_quantize_for_matmul and not dtype_dict[weights_dtype]["is_packed"]:
            result_shape = weight.shape
            weight = weight.flatten(1,-1)
            reduction_axes = -1
    elif layer_class_name in conv_transpose_types:
        is_conv_transpose_type = True
        reduction_axes = 0
        channel_size, output_channel_size = weight.shape[:2]
        use_quantized_matmul = False
    elif layer_class_name in linear_types:
        is_linear_type = True
        reduction_axes = -1
        output_channel_size, channel_size = weight.shape
        if use_quantized_matmul:
            use_quantized_matmul = channel_size >= 32 and output_channel_size >= 32
            use_quantized_matmul = use_quantized_matmul and output_channel_size % 16 == 0 and channel_size % 16 == 0
    else:
        if weight.ndim > 1:
            output_channel_size, channel_size = weight.shape[-2:]
        else:
            output_channel_size, channel_size = 1, weight.shape[-1]
        reduction_axes = -1
        use_quantized_matmul = False

    if use_svd:
        try:
            weight, svd_up, svd_down = apply_svdquant(weight, rank=svd_rank, niter=svd_steps)
            if use_quantized_matmul:
                svd_up = svd_up.t_()
                svd_down = svd_down.t_()
            svd_up, svd_down = prepare_svd_for_matmul(svd_up, svd_down, use_quantized_matmul, use_contiguous_mm)
        except Exception:
            svd_up, svd_down = None, None
    else:
        svd_up, svd_down = None, None

    if group_size == 0:
        if use_quantized_matmul and not re_quantize_for_matmul and dtype_dict[weights_dtype]["num_bits"] >= 6:
            group_size = -1
        elif is_linear_type:
            group_size = 2 ** ((3 if (svd_up is not None or using_pre_calculated_svd) else 2) + dtype_dict[weights_dtype]["num_bits"])
        else:
            group_size = 2 ** ((2 if (svd_up is not None or using_pre_calculated_svd) else 1) + dtype_dict[weights_dtype]["num_bits"])

    if group_size > 0:
        if group_size >= channel_size:
            group_size = channel_size
            num_of_groups = 1
        else:
            num_of_groups = channel_size // group_size
            while num_of_groups * group_size != channel_size:
                num_of_groups -= 1
                if num_of_groups <= 1:
                    group_size = channel_size
                    num_of_groups = 1
                    break
                group_size = channel_size // num_of_groups
        group_size = int(group_size)
        num_of_groups = int(num_of_groups)

        if num_of_groups > 1:
            if result_shape is None:
                result_shape = weight.shape
            new_shape = list(result_shape)
            if is_conv_type:
                new_shape[1] = group_size
                new_shape.insert(1, num_of_groups)
                reduction_axes = 2
            elif is_conv_transpose_type:
                new_shape[0] = group_size
                new_shape.insert(0, num_of_groups)
                reduction_axes = 1
            else:
                last_dim_index = weight.ndim
                new_shape[last_dim_index - 1 : last_dim_index] = (num_of_groups, group_size)
            weight = weight.reshape(new_shape)
        else:
            group_size = -1

    quantized_weight, scale, zero_point = quantize_weight(weight, reduction_axes, weights_dtype, use_stochastic_rounding=use_stochastic_rounding)
    
    if (
        not dequantize_fp32
        and dtype_dict[weights_dtype]["num_bits"] <= 8
        and not (
            use_quantized_matmul
            and not dtype_dict[quantized_matmul_dtype]["is_integer"]
            and (not use_tensorwise_fp8_matmul or dtype_dict[quantized_matmul_dtype]["num_bits"] == 16)
        )
    ):
        scale = scale.to(dtype=torch_dtype)
        if zero_point is not None:
            zero_point = zero_point.to(dtype=torch_dtype)
        if svd_up is not None:
            svd_up = svd_up.to(dtype=torch_dtype)
            svd_down = svd_down.to(dtype=torch_dtype)

    re_quantize_for_matmul = re_quantize_for_matmul or num_of_groups > 1
    if use_quantized_matmul and not re_quantize_for_matmul and not dtype_dict[weights_dtype]["is_packed"]:
        scale.t_()
        quantized_weight.t_()
        quantized_weight = prepare_weight_for_matmul(quantized_weight, use_contiguous_mm)
        if not use_tensorwise_fp8_matmul and not dtype_dict[quantized_matmul_dtype]["is_integer"]:
            scale = scale.to(dtype=torch.float32)

    if dtype_dict[weights_dtype]["is_packed"]:
        if dtype_dict[weights_dtype]["is_integer"]:
            if dtype_dict[weights_dtype]["is_unsigned"]:
                quantized_weight = pack_int_asymetric(quantized_weight, weights_dtype)
            else:
                quantized_weight = pack_int_symetric(quantized_weight, weights_dtype)
        else:
            quantized_weight = pack_float(quantized_weight, weights_dtype)
    else:
        quantized_weight = quantized_weight.to(dtype=dtype_dict[weights_dtype]["torch_dtype"])

    info = {
        "weights_dtype": weights_dtype,
        "quantized_matmul_dtype": quantized_matmul_dtype,
        "group_size": group_size,
        "use_svd": use_svd,
        "svd_rank": svd_rank,
        "use_quantized_matmul": use_quantized_matmul,
        "use_stochastic_rounding": use_stochastic_rounding,
        "dequantize_fp32": dequantize_fp32,
        "use_tensorwise_fp8_matmul": use_tensorwise_fp8_matmul,
        "use_contiguous_mm": use_contiguous_mm
    }

    return quantized_weight, scale, zero_point, svd_up, svd_down, info
