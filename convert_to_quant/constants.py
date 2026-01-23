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
    "nvfp4",  # NVIDIA FP4 E2M1 block quantization
    "mxfp8",  # Microscaling FP8 block quantization
    "hybrid_mxfp8",  # Hybrid MXFP8 (MXFP8 + tensorwise fallback)
}

# Global config: normalize 1-element scale arrays to scalars (set from CLI)
NORMALIZE_SCALES_ENABLED = True
