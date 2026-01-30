"""
Constants and configuration values for convert_to_quant.

Contains model-specific key name filters and dtype settings.
"""
import torch

# --- Model-specific exclusion lists (layers to skip quantization) ---
AVOID_KEY_NAMES = [
    "norm",
    "bias",
    "embed_tokens",
    "lm_head",
    "shared",
    "patch_embedding",
    "audio_model.patch_embedding",
    "ref_conv",
    "control_adapter",
    "motion_encoder.enc.net_app",
    "face_encoder.conv",
    "pose_patch_embedding",
    "motion_encoder.enc.fc",
    "img_emb.proj",
    "k_norm",
    "q_norm",
    "motion_encoder.dec",
    "head.modulation",
    "casual_audio_encoder",
    "cond_encoder",
    "frame_packer",
    "norm_k",
    "norm_q",
    "tekken_model",
    "multi_modal_projector",
    "patch_conv",
    "ln_pre",
    "input_layernorm",
    "attention_norm",
    "post_attention_layernorm",
]
T5XXL_REMOVE_KEY_NAMES = ["decoder", "lm_head"]
VISUAL_AVOID_KEY_NAMES = ["mlp.down_proj", "mlp.up_proj", "mlp.gate_proj"]
QWEN_AVOID_KEY_NAMES = ["norm_added_k", "norm_added_q", "norm_k", "norm_q", "txt_norm"]
HUNYUAN_AVOID_KEY_NAMES = [
    "layernorm",
    "img_attn_k_norm",
    "img_attn_q_norm",
    "txt_attn_k_norm",
    "txt_attn_q_norm",
    "norm1",
    "norm2",
    "vision_in.proj.0",
    "vision_in.proj.4",
    "img_in.proj",
    "cond_type_embedding",
]
ZIMAGE_AVOID_KEY_NAMES = [
    "cap_embedder.0",
    "cap_pad_token",
    "attention_norm1",
    "attention_norm2",
    "ffn_norm1",
    "ffn_norm2",
    "k_norm",
    "q_norm",
    "x_pad_token",
]

# --- Layer key names for specific models (layers to include as high-precision) ---
FLUX2_LAYER_KEYNAMES = [
    "stream_modulation",
    "guidance_in",
    "time_in",
    "final_layer",
    "img_in",
    "txt_in",
]
DISTILL_LAYER_KEYNAMES_LARGE = [
    "distilled_guidance_layer",
    "final_layer",
    "img_in",
    "txt_in",
]
DISTILL_LAYER_KEYNAMES_SMALL = ["distilled_guidance_layer"]
NERF_LAYER_KEYNAMES_LARGE = [
    "distilled_guidance_layer",
    "nerf_blocks",
    "nerf_image_embedder",
    "txt_in",
]
NERF_LAYER_KEYNAMES_SMALL = [
    "distilled_guidance_layer",
    "nerf_blocks",
    "nerf_image_embedder",
]
RADIANCE_LAYER_KEYNAMES = ["img_in_patch", "nerf_final_layer_conv", "__x0__"]
WAN_LAYER_KEYNAMES = [
    "text_embedding",
    "time_embedding",
    "audio_model.text_embedding",
    "casual_audio_encoder",
    "frame_packer",
    "trainable_cond_mask",
    "cond_encoder",
    "audio_model.time_embedding",
    "time_projection",
    "video_model.time_projection",
    "head.head",
    "face_encoder.out_proj",
    "face_adapter",
    "audio_injector",
]
QWEN_LAYER_KEYNAMES = [
    "time_text_embed",
    "img_in",
    "norm_out",
    "proj_out",
    "transformer_blocks.0.img_mod.1",
    "txt_in",
]
ZIMAGE_LAYER_KEYNAMES = [
    "x_embedder",
    "clip_text_pooled_proj",
    "final_layer",
    "cap_embedder.1",
    "adaLN_modulation",
    "t_embedder",
    "time_text_embed",
]
ZIMAGE_REFINER_LAYER_KEYNAMES = ["context_refiner", "noise_refiner"]

# --- Model Filter Registry ---
# Each entry maps a CLI flag (--radiance, --flux2, etc.) to its layer patterns.
# Keys:
#   "help"     - Help text for CLI
#   "category" - For grouping in --help-filters (text, diffusion, video, image)
#   "exclude"  - Layers to skip quantization entirely (extends AVOID_KEY_NAMES)
#   "highprec" - Layers to keep in high precision (not quantized)
#   "remove"   - Layers to remove from output entirely (rare, e.g., T5XXL decoder)

MODEL_FILTERS = {
    # Text Encoders
    "t5xxl": {
        "help": "T5-XXL text encoder: skip norms/biases, remove decoder layers",
        "category": "text",
        "exclude": AVOID_KEY_NAMES,
        "remove": T5XXL_REMOVE_KEY_NAMES,
    },
    "mistral": {
        "help": "Mistral text encoder exclusions",
        "category": "text",
        "exclude": AVOID_KEY_NAMES,
    },
    "visual": {
        "help": "Visual encoder: skip MLP layers (down/up/gate proj)",
        "category": "text",
        "exclude": VISUAL_AVOID_KEY_NAMES,
    },
    # Diffusion Models (Flux-style)
    "flux2": {
        "help": "Flux.2: keep modulation/guidance/time/final layers high-precision",
        "category": "diffusion",
        "highprec": FLUX2_LAYER_KEYNAMES,
    },
    "distillation_large": {
        "help": "Chroma/distilled (large): keep distilled_guidance, final, img/txt_in high-precision",
        "category": "diffusion",
        "highprec": DISTILL_LAYER_KEYNAMES_LARGE,
    },
    "distillation_small": {
        "help": "Chroma/distilled (small): keep only distilled_guidance high-precision",
        "category": "diffusion",
        "highprec": DISTILL_LAYER_KEYNAMES_SMALL,
    },
    "nerf_large": {
        "help": "NeRF (large): keep nerf_blocks, distilled_guidance, txt_in high-precision",
        "category": "diffusion",
        "highprec": NERF_LAYER_KEYNAMES_LARGE,
    },
    "nerf_small": {
        "help": "NeRF (small): keep nerf_blocks, distilled_guidance high-precision",
        "category": "diffusion",
        "highprec": NERF_LAYER_KEYNAMES_SMALL,
    },
    "radiance": {
        "help": "Radiance model: keep img_in_patch, nerf_final_layer high-precision",
        "category": "diffusion",
        "highprec": RADIANCE_LAYER_KEYNAMES,
    },
    # Video Models
    "wan": {
        "help": "WAN video model: skip embeddings, encoders, head",
        "category": "video",
        "exclude": AVOID_KEY_NAMES,
        "highprec": WAN_LAYER_KEYNAMES,
    },
    "hunyuan": {
        "help": "Hunyuan Video 1.5: skip layernorm, attn norms, vision_in",
        "category": "video",
        "exclude": HUNYUAN_AVOID_KEY_NAMES,
    },
    # Image Models
    "qwen": {
        "help": "Qwen Image: skip added norms, keep time_text_embed high-precision",
        "category": "image",
        "exclude": QWEN_AVOID_KEY_NAMES,
        "highprec": QWEN_LAYER_KEYNAMES,
    },
    "zimage": {
        "help": "Z-Image: skip cap_embedder/norms, keep x_embedder/final high-precision",
        "category": "image",
        "exclude": ZIMAGE_AVOID_KEY_NAMES,
        "highprec": ZIMAGE_LAYER_KEYNAMES,
    },
    "zimage_refiner": {
        "help": "Z-Image Refiner: keep context/noise refiner high-precision",
        "category": "image",
        "exclude": ZIMAGE_AVOID_KEY_NAMES,
        "highprec": ZIMAGE_REFINER_LAYER_KEYNAMES,
    },
}


def build_exclusion_patterns(active_filters: dict) -> tuple:
    """
    Build layer exclusion patterns from active filter flags.

    Args:
        active_filters: Dict of filter_name -> bool (e.g., {"radiance": True, "t5xxl": False})

    Returns:
        Tuple of (exclude_patterns, highprec_patterns, remove_patterns)
    """
    exclude = []
    highprec = []
    remove = []

    for name, cfg in MODEL_FILTERS.items():
        if active_filters.get(name, False):
            exclude.extend(cfg.get("exclude", []))
            highprec.extend(cfg.get("highprec", []))
            remove.extend(cfg.get("remove", []))

    return exclude, highprec, remove

# --- Dtype settings ---
TARGET_FP8_DTYPE = torch.float8_e4m3fn
TARGET_INT8_DTYPE = torch.int8
COMPUTE_DTYPE = torch.float32
SCALE_DTYPE = torch.float32

# FP8 constants
FP8_MIN = float(torch.finfo(TARGET_FP8_DTYPE).min)
FP8_MAX = float(torch.finfo(TARGET_FP8_DTYPE).max)
FP8_MIN_POS = float(torch.finfo(TARGET_FP8_DTYPE).tiny)

# INT8 constants (using symmetric range [-127, 127] for symmetric quantization)
INT8_MIN = int(torch.iinfo(TARGET_INT8_DTYPE).min)  # -128
INT8_MAX = int(torch.iinfo(TARGET_INT8_DTYPE).max)  # 127
INT8_SYMMETRIC_MAX = min(abs(INT8_MIN), INT8_MAX)  # 127 (symmetric range)

# FP4 NVFP4 E2M1 constants
TARGET_FP4_DTYPE = torch.uint8  # Packed format (2 values per byte)
FP4_E2M1_MAX = 6.0
FP4_E2M1_EPS = 0.5
FP4_BLOCK_SIZE = 16  # NVFP4 uses 16-element blocks

# MXFP8 (Microscaling FP8) constants
# MXFP8 uses FP8 E4M3 data with E8M0 (power-of-2 exponent) block scales
MXFP8_BLOCK_SIZE = 32  # MXFP8 uses 32-element blocks
MXFP8_DTYPE = torch.float8_e4m3fn  # Data stored as FP8 E4M3
E8M0_BIAS = 127  # Exponent bias for E8M0 format (value = 2^(exp - 127))

# --- Adaptive LR Tier Configuration ---
# Used by 'original' optimizer in LearnedRoundingConverter and LearnedNVFP4Converter.
# Format: List of (counter_threshold, improvement_mult, decay_mult, min_lr)
#   - counter_threshold: worse_loss_counter must be >= this to use this tier
#   - improvement_mult: LR multiplier when loss improves (boost)
#   - decay_mult: LR multiplier when loss worsens (decay)
#   - min_lr: minimum LR floor for decay operations at this tier
ADAPTIVE_LR_TIERS_IMPROVE = [
    # (counter_threshold, multiplier, max_lr)
    (0, 1.25, 100.0),     # counter < 50: boost by 1.25x
    (50, 1.375, 100.0),   # 50 <= counter < 75
    (75, 1.5, 100.0),     # 75 <= counter < 100
    (100, 1.75, 100.0),   # 100 <= counter < 125
    (125, 2.0, 100.0),    # 125 <= counter < 150
    (150, 2.25, 100.0),   # 150 <= counter < 200
    (200, 2.5, 100.0),    # 200 <= counter < 250
    (250, 2.75, 100.0),   # 250 <= counter < 300
    (300, 3.0, 100.0),    # counter >= 300
]

ADAPTIVE_LR_TIERS_DECAY = [
    # (counter_threshold, multiplier, min_lr)
    (0, 0.95, 9e-8),      # counter < 26: decay by 0.95x
    (26, 0.97, 8e-8),     # 26 <= counter < 51
    (51, 0.985, 7e-8),    # 51 <= counter < 76
    (76, 0.9875, 6e-8),   # 76 <= counter < 101
    (101, 0.98875, 5e-8), # 101 <= counter < 151
    (151, 0.99, 4e-8),    # 151 <= counter < 201
    (201, 0.99125, 3e-8), # 201 <= counter < 251
    (251, 0.9925, 2e-8),  # 251 <= counter < 301
    (301, 0.995, 5e-9),   # counter >= 301
]

# Valid quantization formats (maps to QUANT_ALGOS in quant_ops.py)
VALID_QUANT_FORMATS = {
    "float8_e4m3fn",
    "float8_e4m3fn_rowwise",
    "float8_e4m3fn_blockwise",
    "float8_e4m3fn_block3d",
    "int8_blockwise",
    "int8_tensorwise",
    "nvfp4",  # NVIDIA FP4 E2M1 block quantization
    "mxfp8",  # Microscaling FP8 block quantization
    "hybrid_mxfp8",  # Hybrid MXFP8 (MXFP8 + tensorwise fallback)
    "sdnq",         # SDNQ Stochastic Differentiable Neural Quantization
}

# --- SDNQ Specific Constants (moved from converters/constants.py) ---
SDNQ_DTYPE_DICT = {
    ### Integers
    "int32": {"min": -2147483648, "max": 2147483647, "num_bits": 32, "sign": 1, "exponent": 0, "mantissa": 31, "target_dtype": torch.int32, "torch_dtype": torch.int32, "storage_dtype": torch.int32, "is_unsigned": False, "is_integer": True, "is_packed": False},
    "int16": {"min": -32768, "max": 32767, "num_bits": 16, "sign": 1, "exponent": 0, "mantissa": 15, "target_dtype": torch.int16, "torch_dtype": torch.int16, "storage_dtype": torch.int16, "is_unsigned": False, "is_integer": True, "is_packed": False},
    "int8": {"min": -128, "max": 127, "num_bits": 8, "sign": 1, "exponent": 0, "mantissa": 7, "target_dtype": torch.int8, "torch_dtype": torch.int8, "storage_dtype": torch.int8, "is_unsigned": False, "is_integer": True, "is_packed": False},
    ### Custom Integers
    "int7": {"min": -64, "max": 63, "num_bits": 7, "sign": 1, "exponent": 0, "mantissa": 6, "target_dtype": "int7", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int6": {"min": -32, "max": 31, "num_bits": 6, "sign": 1, "exponent": 0, "mantissa": 5, "target_dtype": "int6", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int5": {"min": -16, "max": 15, "num_bits": 5, "sign": 1, "exponent": 0, "mantissa": 4, "target_dtype": "int5", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int4": {"min": -8, "max": 7, "num_bits": 4, "sign": 1, "exponent": 0, "mantissa": 3, "target_dtype": "int4", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int3": {"min": -4, "max": 3, "num_bits": 3, "sign": 1, "exponent": 0, "mantissa": 2, "target_dtype": "int3", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int2": {"min": -2, "max": 1, "num_bits": 2, "sign": 1, "exponent": 0, "mantissa": 1, "target_dtype": "int2", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    ### Unsigned Integers
    "uint32": {"min": 0, "max": 4294967295, "num_bits": 32, "sign": 0, "exponent": 0, "mantissa": 32, "target_dtype": torch.uint32, "torch_dtype": torch.uint32, "storage_dtype": torch.uint32, "is_unsigned": True, "is_integer": True, "is_packed": False},
    "uint16": {"min": 0, "max": 65535, "num_bits": 16, "sign": 0, "exponent": 0, "mantissa": 16, "target_dtype": torch.uint16, "torch_dtype": torch.uint16, "storage_dtype": torch.uint16, "is_unsigned": True, "is_integer": True, "is_packed": False},
    "uint8": {"min": 0, "max": 255, "num_bits": 8, "sign": 0, "exponent": 0, "mantissa": 8, "target_dtype": torch.uint8, "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": False},
    ### Custom Unsigned Integers
    "uint7": {"min": 0, "max": 127, "num_bits": 7, "sign": 0, "exponent": 0, "mantissa": 7, "target_dtype": "uint7", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint6": {"min": 0, "max": 63, "num_bits": 6, "sign": 0, "exponent": 0, "mantissa": 6, "target_dtype": "uint6", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint5": {"min": 0, "max": 31, "num_bits": 5, "sign": 0, "exponent": 0, "mantissa": 5, "target_dtype": "uint5", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint4": {"min": 0, "max": 15, "num_bits": 4, "sign": 0, "exponent": 0, "mantissa": 4, "target_dtype": "uint4", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint3": {"min": 0, "max": 7, "num_bits": 3, "sign": 0, "exponent": 0, "mantissa": 3, "target_dtype": "uint3", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint2": {"min": 0, "max": 3, "num_bits": 2, "sign": 0, "exponent": 0, "mantissa": 2, "target_dtype": "uint2", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint1": {"min": 0, "max": 1, "num_bits": 1, "sign": 0, "exponent": 0, "mantissa": 1, "target_dtype": torch.bool, "torch_dtype": torch.bool, "storage_dtype": torch.bool, "is_unsigned": True, "is_integer": True, "is_packed": True},
    ### Floats
    "float32": {"min": -3.40282e+38, "max": 3.40282e+38, "num_bits": 32, "sign": 1, "exponent": 8, "mantissa": 23, "target_dtype": torch.float32, "torch_dtype": torch.float32, "storage_dtype": torch.float32, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "bfloat16": {"min": -3.38953e+38, "max": 3.38953e+38, "num_bits": 16, "sign": 1, "exponent": 8, "mantissa": 7, "target_dtype": torch.bfloat16, "torch_dtype": torch.bfloat16, "storage_dtype": torch.bfloat16, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "float16": {"min": -65504.0, "max": 65504.0, "num_bits": 16, "sign": 1, "exponent": 5, "mantissa": 10, "target_dtype": torch.float16, "torch_dtype": torch.float16, "storage_dtype": torch.float16, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "float8_e4m3fn": {"min": -448.0, "max": 448.0, "num_bits": 8, "sign": 1, "exponent": 4, "mantissa": 3, "target_dtype": torch.float8_e4m3fn, "torch_dtype": torch.float8_e4m3fn, "storage_dtype": torch.float8_e4m3fn, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "float8_e5m2": {"min": -57344.0, "max": 57344.0, "num_bits": 8, "sign": 1, "exponent": 5, "mantissa": 2, "target_dtype": torch.float8_e5m2, "torch_dtype": torch.float8_e5m2, "storage_dtype": torch.float8_e5m2, "is_unsigned": False, "is_integer": False, "is_packed": False},
    ### Custom Floats
    "float16_e1m14fn": {"min": -3.9998779296875, "max": 3.9998779296875, "num_bits": 16, "sign": 1, "exponent": 1, "mantissa": 14, "min_normal": 1.00006103515625, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float16_e2m13fn": {"min": -7.99951171875, "max": 7.99951171875, "num_bits": 16, "sign": 1, "exponent": 2, "mantissa": 13, "min_normal": 0.50006103515625, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float16_e3m12fn": {"min": -31.99609375, "max": 31.99609375, "num_bits": 16, "sign": 1, "exponent": 3, "mantissa": 12, "min_normal": 0.125030517578125, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float16_e4m11fn": {"min": -511.875, "max": 511.875, "num_bits": 16, "sign": 1, "exponent": 4, "mantissa": 11, "min_normal": 0.007816314697265625, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float8_e1m6fn": {"min": -3.96875, "max": 3.96875, "num_bits": 8, "sign": 1, "exponent": 1, "mantissa": 6, "min_normal": 1.015625, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float8_e2m5fn": {"min": -7.875, "max": 7.875, "num_bits": 8, "sign": 1, "exponent": 2, "mantissa": 5, "min_normal": 0.515625, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float8_e3m4fn": {"min": -31.0, "max": 31.0, "num_bits": 8, "sign": 1, "exponent": 3, "mantissa": 4, "min_normal": 0.1328125, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float7_e1m5fn": {"min": -3.9375, "max": 3.9375, "num_bits": 7, "sign": 1, "exponent": 1, "mantissa": 5, "min_normal": 1.03125, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float7_e2m4fn": {"min": -7.75, "max": 7.75, "num_bits": 7, "sign": 1, "exponent": 2, "mantissa": 4, "min_normal": 0.53125, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float7_e3m3fn": {"min": -30.0, "max": 30.0, "num_bits": 7, "sign": 1, "exponent": 3, "mantissa": 3, "min_normal": 0.140625, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float7_e4m2fn": {"min": -448.0, "max": 448.0, "num_bits": 7, "sign": 1, "exponent": 4, "mantissa": 2, "min_normal": 0.009765625, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float7_e5m1fn": {"min": -98304.0, "max": 98304.0, "num_bits": 7, "sign": 1, "exponent": 5, "mantissa": 1, "min_normal": 4.57763671875e-05, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float6_e1m4fn": {"min": -3.875, "max": 3.875, "num_bits": 6, "sign": 1, "exponent": 1, "mantissa": 4, "min_normal": 1.0625, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float6_e2m3fn": {"min": -7.5, "max": 7.5, "num_bits": 6, "sign": 1, "exponent": 2, "mantissa": 3, "min_normal": 0.5625, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float6_e3m2fn": {"min": -28.0, "max": 28.0, "num_bits": 6, "sign": 1, "exponent": 3, "mantissa": 2, "min_normal": 0.15625, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float6_e4m1fn": {"min": -384.0, "max": 384.0, "num_bits": 6, "sign": 1, "exponent": 4, "mantissa": 1, "min_normal": 0.01171875, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float6_e5m0fn": {"min": -65536.0, "max": 65536.0, "num_bits": 6, "sign": 1, "exponent": 5, "mantissa": 0, "min_normal": 6.103515625e-05, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float5_e1m3fn": {"min": -3.75, "max": 3.75, "num_bits": 5, "sign": 1, "exponent": 1, "mantissa": 3, "min_normal": 1.125, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float5_e2m2fn": {"min": -7.0, "max": 7.0, "num_bits": 5, "sign": 1, "exponent": 2, "mantissa": 2, "min_normal": 0.625, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float5_e3m1fn": {"min": -24.0, "max": 24.0, "num_bits": 5, "sign": 1, "exponent": 3, "mantissa": 1, "min_normal": 0.1875, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float5_e4m0fn": {"min": -256.0, "max": 256.0, "num_bits": 5, "sign": 1, "exponent": 4, "mantissa": 0, "min_normal": 0.015625, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float4_e1m2fn": {"min": -3.5, "max": 3.5, "num_bits": 4, "sign": 1, "exponent": 1, "mantissa": 2, "min_normal": 1.25, "target_dtype": "fp4", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float4_e2m1fn": {"min": -6.0, "max": 6.0, "num_bits": 4, "sign": 1, "exponent": 2, "mantissa": 1, "min_normal": 0.75, "target_dtype": "fp4", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float4_e3m0fn": {"min": -16.0, "max": 16.0, "num_bits": 4, "sign": 1, "exponent": 3, "mantissa": 0, "min_normal": 0.25, "target_dtype": "fp4", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float3_e1m1fn": {"min": -3.0, "max": 3.0, "num_bits": 3, "sign": 1, "exponent": 1, "mantissa": 1, "min_normal": 1.5, "target_dtype": "fp3", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float3_e2m0fn": {"min": -4.0, "max": 4.0, "num_bits": 3, "sign": 1, "exponent": 2, "mantissa": 0, "min_normal": 1.0, "target_dtype": "fp3", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float2_e1m0fn": {"min": -2.0, "max": 2.0, "num_bits": 2, "sign": 1, "exponent": 1, "mantissa": 0, "min_normal": 2.0, "target_dtype": "fp2", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    ### Custom Unsigned Floats
    "float16_e1m15fnu": {"min": 0, "max": 3.99993896484375, "num_bits": 16, "sign": 0, "exponent": 1, "mantissa": 15, "min_normal": 1.000030517578125, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float16_e2m14fnu": {"min": 0, "max": 7.999755859375, "num_bits": 16, "sign": 0, "exponent": 2, "mantissa": 14, "min_normal": 0.500030517578125, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float16_e3m13fnu": {"min": 0, "max": 31.998046875, "num_bits": 16, "sign": 0, "exponent": 3, "mantissa": 13, "min_normal": 0.1250152587890625, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float16_e4m12fnu": {"min": 0, "max": 511.9375, "num_bits": 16, "sign": 0, "exponent": 4, "mantissa": 12, "min_normal": 0.007814407348632812, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float16_e5m11fnu": {"min": 0, "max": 131040.0, "num_bits": 16, "sign": 0, "exponent": 5, "mantissa": 11, "min_normal": 3.053247928619385e-05, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float8_e1m7fnu": {"min": 0, "max": 3.984375, "num_bits": 8, "sign": 0, "exponent": 1, "mantissa": 7, "min_normal": 1.0078125, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float8_e2m6fnu": {"min": 0, "max": 7.9375, "num_bits": 8, "sign": 0, "exponent": 2, "mantissa": 6, "min_normal": 0.5078125, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float8_e3m5fnu": {"min": 0, "max": 31.5, "num_bits": 8, "sign": 0, "exponent": 3, "mantissa": 5, "min_normal": 0.12890625, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float8_e4m4fnu": {"min": 0, "max": 496.0, "num_bits": 8, "sign": 0, "exponent": 4, "mantissa": 4, "min_normal": 0.00830078125, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float8_e5m3fnu": {"min": 0, "max": 122880.0, "num_bits": 8, "sign": 0, "exponent": 5, "mantissa": 3, "min_normal": 3.4332275390625e-05, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float7_e1m6fnu": {"min": 0, "max": 3.96875, "num_bits": 7, "sign": 0, "exponent": 1, "mantissa": 6, "min_normal": 1.015625, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float7_e2m5fnu": {"min": 0, "max": 7.875, "num_bits": 7, "sign": 0, "exponent": 2, "mantissa": 5, "min_normal": 0.515625, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float7_e3m4fnu": {"min": 0, "max": 31.0, "num_bits": 7, "sign": 0, "exponent": 3, "mantissa": 4, "min_normal": 0.1328125, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float7_e4m3fnu": {"min": 0, "max": 480.0, "num_bits": 7, "sign": 0, "exponent": 4, "mantissa": 3, "min_normal": 0.0087890625, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float7_e5m2fnu": {"min": 0, "max": 114688.0, "num_bits": 7, "sign": 0, "exponent": 5, "mantissa": 2, "min_normal": 3.814697265625e-05, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float6_e1m5fnu": {"min": 0, "max": 3.9375, "num_bits": 6, "sign": 0, "exponent": 1, "mantissa": 5, "min_normal": 1.03125, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float6_e2m4fnu": {"min": 0, "max": 7.75, "num_bits": 6, "sign": 0, "exponent": 2, "mantissa": 4, "min_normal": 0.53125, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float6_e3m3fnu": {"min": 0, "max": 30.0, "num_bits": 6, "sign": 0, "exponent": 3, "mantissa": 3, "min_normal": 0.140625, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float6_e4m2fnu": {"min": 0, "max": 448.0, "num_bits": 6, "sign": 0, "exponent": 4, "mantissa": 2, "min_normal": 0.009765625, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float6_e5m1fnu": {"min": 0, "max": 98304.0, "num_bits": 6, "sign": 0, "exponent": 5, "mantissa": 1, "min_normal": 4.57763671875e-05, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float5_e1m4fnu": {"min": 0, "max": 3.875, "num_bits": 5, "sign": 0, "exponent": 1, "mantissa": 4, "min_normal": 1.0625, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float5_e2m3fnu": {"min": 0, "max": 7.5, "num_bits": 5, "sign": 0, "exponent": 2, "mantissa": 3, "min_normal": 0.5625, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float5_e3m2fnu": {"min": 0, "max": 28.0, "num_bits": 5, "sign": 0, "exponent": 3, "mantissa": 2, "min_normal": 0.15625, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float5_e4m1fnu": {"min": 0, "max": 384.0, "num_bits": 5, "sign": 0, "exponent": 4, "mantissa": 1, "min_normal": 0.01171875, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float5_e5m0fnu": {"min": 0, "max": 65536.0, "num_bits": 5, "sign": 0, "exponent": 5, "mantissa": 0, "min_normal": 6.103515625e-05, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float4_e1m3fnu": {"min": 0, "max": 3.75, "num_bits": 4, "sign": 0, "exponent": 1, "mantissa": 3, "min_normal": 1.125, "target_dtype": "fp4", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float4_e2m2fnu": {"min": 0, "max": 7.0, "num_bits": 4, "sign": 0, "exponent": 2, "mantissa": 2, "min_normal": 0.625, "target_dtype": "fp4", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float4_e3m1fnu": {"min": 0, "max": 24.0, "num_bits": 4, "sign": 0, "exponent": 3, "mantissa": 1, "min_normal": 0.1875, "target_dtype": "fp4", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float4_e4m0fnu": {"min": 0, "max": 256.0, "num_bits": 4, "sign": 0, "exponent": 4, "mantissa": 0, "min_normal": 0.015625, "target_dtype": "fp4", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float3_e1m2fnu": {"min": 0, "max": 3.5, "num_bits": 3, "sign": 0, "exponent": 1, "mantissa": 2, "min_normal": 1.25, "target_dtype": "fp3", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float3_e2m1fnu": {"min": 0, "max": 6.0, "num_bits": 3, "sign": 0, "exponent": 2, "mantissa": 1, "min_normal": 0.75, "target_dtype": "fp3", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float3_e3m0fnu": {"min": 0, "max": 16.0, "num_bits": 3, "sign": 0, "exponent": 3, "mantissa": 0, "min_normal": 0.25, "target_dtype": "fp3", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float2_e1m1fnu": {"min": 0, "max": 3.0, "num_bits": 2, "sign": 0, "exponent": 1, "mantissa": 1, "min_normal": 1.5, "target_dtype": "fp2", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float2_e2m0fnu": {"min": 0, "max": 4.0, "num_bits": 2, "sign": 0, "exponent": 2, "mantissa": 0, "min_normal": 1.0, "target_dtype": "fp2", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float1_e1m0fnu": {"min": 0, "max": 2.0, "num_bits": 1, "sign": 0, "exponent": 1, "mantissa": 0, "min_normal": 2.0, "target_dtype": "fp1", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
}

SDNQ_DTYPE_DICT["fp32"] = SDNQ_DTYPE_DICT["float32"]
SDNQ_DTYPE_DICT["bf16"] = SDNQ_DTYPE_DICT["bfloat16"]
SDNQ_DTYPE_DICT["fp16"] = SDNQ_DTYPE_DICT["float16"]
SDNQ_DTYPE_DICT["fp8"] = SDNQ_DTYPE_DICT["float8_e4m3fn"]
SDNQ_DTYPE_DICT["fp7"] = SDNQ_DTYPE_DICT["float7_e3m3fn"]
SDNQ_DTYPE_DICT["fp6"] = SDNQ_DTYPE_DICT["float6_e3m2fn"]
SDNQ_DTYPE_DICT["fp5"] = SDNQ_DTYPE_DICT["float5_e2m2fn"]
SDNQ_DTYPE_DICT["fp4"] = SDNQ_DTYPE_DICT["float4_e2m1fn"]
SDNQ_DTYPE_DICT["fp3"] = SDNQ_DTYPE_DICT["float3_e1m1fn"]
SDNQ_DTYPE_DICT["fp2"] = SDNQ_DTYPE_DICT["float2_e1m0fn"]
SDNQ_DTYPE_DICT["fp1"] = SDNQ_DTYPE_DICT["float1_e1m0fnu"]
SDNQ_DTYPE_DICT["bool"] = SDNQ_DTYPE_DICT["uint1"]
SDNQ_DTYPE_DICT["int1"] = SDNQ_DTYPE_DICT["uint1"]

SDNQ_LINEAR_TYPES = {"Linear", "SDNQLinear"}
SDNQ_CONV_TYPES = {"Conv1d", "Conv2d", "Conv3d", "SDNQConv1d", "SDNQConv2d", "SDNQConv3d"}
SDNQ_CONV_TRANSPOSE_TYPES = {"ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "SDNQConvTranspose1d", "SDNQConvTranspose2d", "SDNQConvTranspose3d"}
SDNQ_ALLOWED_TYPES = set.union(SDNQ_LINEAR_TYPES, SDNQ_CONV_TYPES, SDNQ_CONV_TRANSPOSE_TYPES)

# Global config: normalize 1-element scale arrays to scalars (set from CLI)
NORMALIZE_SCALES_ENABLED = True
