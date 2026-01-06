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

# Valid quantization formats (maps to QUANT_ALGOS in quant_ops.py)
VALID_QUANT_FORMATS = {
    "float8_e4m3fn",
    "float8_e4m3fn_rowwise",
    "float8_e4m3fn_blockwise",
    "float8_e4m3fn_block3d",
    "int8_blockwise",
    "nvfp4",  # NVIDIA FP4 E2M1 block quantization
}

# Global config: normalize 1-element scale arrays to scalars (set from CLI)
NORMALIZE_SCALES_ENABLED = True
