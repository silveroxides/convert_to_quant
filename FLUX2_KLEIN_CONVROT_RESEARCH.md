# FLUX.2 Klein Base 9B ConvRot research record

Last measured: 2026-08-17

This note preserves the reasoning and evidence behind the FLUX.2 Klein Base 9B INT8 and compact INT4 presets implemented in `convert_to_quant`. It is intentionally more detailed than a preset description so later experiments can begin from measured results instead of repeating the same search.

## Outcome

Two native ComfyUI checkpoints were selected:

- **INT8 ConvRot HQ:** all 112 eligible transformer projections use native `int8_tensorwise` W8A8 with regular Hadamard ConvRot group 256. All other tensors remain BF16.
- **INT4 ConvRot HQ Compact:** 84 projections use packed `asym_w4a8_int8` W4A8 with local group size 16 and ConvRot group 256; 28 interaction-sensitive projections remain W8A8; all non-projection tensors remain BF16.

The compact route stores 82.31% of eligible transformer weight elements at four bits. Its complete file is 6,298,490,232 bytes, 33.28% smaller than the 9,439,998,472-byte INT8 file. It is not exactly half-sized because the checkpoint deliberately retains BF16 global tensors and W8 safeguards and because packed layouts also store scales, codebooks, and metadata. Even the rejected all-W4/group-16 checkpoint was approximately 5.623 GB rather than half of the complete INT8 file.

The measured quality conclusion is deliberately narrow:

- INT8 ConvRot was closer to BF16 than FP8 scaled on both decoded-image aggregate and final-latent metrics.
- Compact INT4 was substantially closer to BF16 than the official NVFP4 checkpoint while remaining close to NVFP4's memory/speed class.
- Compact INT4 did **not** beat FP8 scaled image quality. It is the space-efficient option, not the quality-first option.

## Architecture and compilation boundary

The compiler accepts the exact Klein Base 9B transformer shape:

- 8 double-stream blocks, indices 0-7.
- 24 single-stream blocks, indices 0-23.
- 112 eligible 2D projections.
- Double-block routes: image/text attention `qkv` and `proj`, plus image/text MLP `0` and `2`.
- Single-block routes: `linear1` and `linear2`.

Strict validation rejects partial fixtures, another Klein size, or an already-quantized checkpoint unless the explicit research override is used. Biases, modulation, embeddings, input/output projections, norms, and every other non-eligible tensor are copied byte-for-byte from BF16.

The implementation lives in:

- `convert_to_quant/formats/flux_klein_convrot.py`
- `convert_to_quant/cli/main.py`
- `convert_to_quant/cli/argument_parser.py`

Routing is separate from the bit-packing code. A schema-v1 JSON policy names an exact default and only the exceptions. This makes a policy auditable, lets research change routes without changing serialization code, and allows the compiler report to record exact layer counts and formats.

## Quantized layouts

### W8A8 route

1. Rotate each eligible weight over regular Hadamard groups of 256.
2. Apply deterministic per-output-row symmetric INT8 scaling.
3. Store native ComfyUI `int8_tensorwise` weights, scales, and `convrot=true` metadata.

An end-to-end holdout chose `absmax` for the final W8 model. Lower local weight reconstruction error did not consistently imply a better denoising trajectory, so the compiler exposes both `absmax` and deterministic MSE clipping but the GUI preset records the measured choice.

### Packed W4A8 route

1. Apply ConvRot group 256.
2. Quantize weights with comfy-kitchen's native `AsymW4A8Int8Layout` using a shared 16-entry codebook.
3. Use local group-16 relative scales, FP8 relative-scale storage, and per-channel scales.
4. Execute with INT8 activations through ComfyUI's native `asym_w4a8_int8` path.

“INT4” therefore describes the packed weight storage. It does not claim W4A4 arithmetic. The legacy native `convrot_w4a4` serializer remains available through `--flux-klein-int4-layout w4a4`, but it was not the winning Blackwell preset in this study.

## Sensitivity workflow

The search was hierarchical because weight reconstruction error alone was a poor predictor of final images.

1. **Create controlled baselines.** Compile all-W8 and all-W4 checkpoints from the same original BF16 Base 9B file. Validate tensor names, shapes, quant metadata, and the exact 112-route count.
2. **Capture real activations.** Run deterministic ComfyUI denoising prompts and compare corresponding BF16/W8/W4 projection and block responses on the activations actually seen by the model.
3. **Measure causal substitutions.** Substitute one complete block or projection route at a time into a fixed policy, then measure final-latent damage relative to BF16. This avoids ranking layers solely by isolated local error.
4. **Search interactions.** Evaluate architectural seed policies, cumulative W8 additions, pruning, BF16 promotions, and 42 fixed-size same-shape W4/W8 swaps. Same-size swaps were important: several individually attractive upgrades became harmful in combination.
5. **Validate complete candidates.** Re-run the entire policy on four held-out 512px, 20-step trajectories with Euler, the `flux2` scheduler, CFG 3.0, and fixed seeds.
6. **Decode at target sizes.** Generate matched BF16, FP8 scaled, INT8, INT4, and official NVFP4 images at 1024, 1280, 1536, and 2048 pixels. Use identical prompt, seed, text encoder, VAE, sampler, and schedule for each row.
7. **Benchmark isolated runtime.** Start a fresh ComfyUI process per model, warm it, then take the median of two timed diffusion runs. Record CUDA peak allocated memory for diffusion separately from VAE decoding.

The installed research drivers are under the parent Musubi application's `tools/` directory:

- `flux_klein_sensitivity.py`: activation-aware projection/block profiling.
- `flux_klein_substitution_sensitivity.py` and `flux_klein_restore_sensitivity.py`: one-at-a-time causal route tests.
- `flux_klein_policy_search.py`, `flux_klein_policy_ablation.py`, `flux_klein_prune_policy.py`, and `flux_klein_expand_block_policies.py`: policy exploration.
- `flux_klein_policy_eval.py` and `flux_klein_w4a8_sweep.py`: complete-policy and group-size holdouts.
- `flux_klein_comfy_benchmark.py`: native ComfyUI images, time, and peak VRAM.
- `flux_klein_report.py`: self-contained HTML/JSON comparison report.

## Important experimental lessons

- **End-to-end trajectories outrank local weight error.** Several layers with a large isolated error did little final damage, while modest local errors accumulated destructively in attention/MLP interactions.
- **Layer sensitivity is conditional.** A “sensitive layer list” is not universal; it depends on the surrounding precision policy, weight layout, scale group, scheduler, and prompt distribution.
- **Protect routes, not entire blocks, when size matters.** Keeping a whole block at W8 discarded too much of the W4 size benefit. Projection-level safeguards produced a better Pareto point.
- **Group size is a real file-size control.** W4A8 group 4 improved latent quality but scale overhead made the candidate much larger. Group 16 was the better compact point.
- **A single resolution can mislead.** The compact route was particularly strong at 1536px but weaker on the 1280px typography case. Aggregate reporting must retain every per-case metric and image.
- **Compare only the same source variant.** A public compact W4 checkpoint from a different source/recipe produced an unusable comparison (about 0.663 LPIPS at 1024px) and was excluded from selection.
- **Do not infer “four bit equals half the whole file.”** Only eligible weight payloads halve relative to W8. BF16 tensors, W8 safeguards, group scales, codebooks, and headers set a higher checkpoint-level floor.
- **Keep screening and confirmation separate.** Short trajectories are useful for ranking many candidates, but the final choice must be rerun with the full sampler schedule and decoded images.

## Candidate tradeoffs

File sizes below are decimal GB and latent error is final-latent relative RMSE against BF16 on the four-prompt, 20-step 512px holdout.

| Candidate | Approx. file size | Latent rRMSE | Decision |
|---|---:|---:|---|
| All W4A8, group 16 | 5.623 GB | 29.563% | Small, but too much trajectory error |
| All W4A8, group 4 | 7.259 GB | 22.990% | Better error, excessive scale overhead |
| Quality-heavy mixed W4/W8, group 4 | 8.131 GB | 13.627% | Good latent score but too close to INT8 size; decoded LPIPS remained about 0.204 |
| **Final compact mixed W4/W8, group 16** | **6.298 GB** | **21.361%** | Best measured compact Pareto point |
| All W8A8 ConvRot | 9.440 GB | 7.485% | Quality-first selected preset |

The final compact point sacrifices some latent score relative to the large group-4 candidate to preserve the user-visible reason for INT4: a materially smaller checkpoint and lower peak VRAM.

## Exact compact W8 safeguards

Every eligible projection not listed here is packed W4A8. These 28 routes stay W8A8:

```text
double_blocks.0.img_attn.qkv
double_blocks.0.txt_mlp.2
double_blocks.1.img_mlp.2
double_blocks.2.img_attn.qkv
double_blocks.2.txt_mlp.2
double_blocks.3.img_attn.qkv
double_blocks.3.txt_mlp.2
double_blocks.4.img_attn.qkv
double_blocks.4.txt_attn.qkv
double_blocks.4.txt_mlp.2
double_blocks.5.img_attn.qkv
double_blocks.5.img_mlp.2
double_blocks.5.txt_attn.qkv
double_blocks.5.txt_mlp.2
double_blocks.6.img_attn.qkv
double_blocks.6.img_mlp.2
double_blocks.6.txt_mlp.2
double_blocks.7.img_attn.qkv
double_blocks.7.txt_attn.qkv
double_blocks.7.txt_mlp.2
single_blocks.0.linear2
single_blocks.1.linear2
single_blocks.2.linear2
single_blocks.5.linear2
single_blocks.15.linear2
single_blocks.19.linear2
single_blocks.22.linear2
single_blocks.23.linear2
```

The authoritative machine-readable policies in the Musubi integration are:

- `model_quantizer_presets/flux2_klein_9b_int8_convrot_hq.json`
- `model_quantizer_presets/flux2_klein_9b_int4_convrot_hq.json`

## Final decoded-image and runtime results

Hardware/software: NVIDIA RTX PRO 6000 Blackwell Server Edition, CUDA 13.0, PyTorch 2.13.0+cu130, ComfyUI 0.33.0, comfy-kitchen 0.2.31. Each aggregate is the arithmetic mean of the four resolution/prompt cases. Time and VRAM are diffusion-only; time is the median of two warmed runs per case.

| Model | File (GB) | Avg. time (s) | Avg. peak VRAM (GiB) | LPIPS ↓ | SSIM ↑ | BF16 closeness ↑ | Latent rRMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 18.157 | 23.270 | 19.534 | 0.0000 | 1.0000 | 100.00 | 0.000% |
| FP8 scaled | 9.433 | 23.506 | 11.409 | 0.0655 | 0.8909 | 91.27 | 12.942% |
| **INT8 ConvRot HQ** | **9.440** | **17.328** | **11.415** | **0.0446** | **0.9323** | **94.38** | **7.485%** |
| **INT4 ConvRot HQ Compact** | **6.298** | **17.768** | **8.494** | **0.2136** | **0.7309** | **75.86** | **21.361%** |
| Official NVFP4 | 5.809 | 14.298 | 8.073 | 0.2715 | 0.6627 | 69.56 | 50.405% |

Closeness is a display aid, not a new perceptual standard:

```text
100 * (0.5 * SSIM + 0.5 * (1 - clamp(LPIPS, 0, 1)))
```

Always inspect LPIPS, SSIM, latent error, and the actual paired images separately. With only four prompt/resolution cases, these values compare this controlled run; they are not population-level confidence estimates.

The compact INT4 per-resolution results demonstrate why individual cases remain visible:

| Resolution | Time (s) | Peak VRAM (GiB) | LPIPS ↓ | SSIM ↑ |
|---:|---:|---:|---:|---:|
| 1024 | 6.880 | 7.167 | 0.1653 | 0.7780 |
| 1280 | 11.154 | 7.791 | 0.4080 | 0.6348 |
| 1536 | 16.991 | 8.544 | 0.0871 | 0.9266 |
| 2048 | 36.046 | 10.473 | 0.1941 | 0.5841 |

## Reproducing compilation

Run compilation with the Musubi environment and benchmarking with the ComfyUI environment. This separation is intentional: the compiler needs its conversion dependencies, while tests must exercise ComfyUI's actual loader and kernels.

```bash
# From the SECourses_Musubi_Trainer root.
./venv/bin/ctq \
  -i /path/to/FLUX-2-Klein-Base-9b.safetensors \
  -o /path/to/FLUX-2-Klein-Base-9b-INT8-ConvRot-HQ.safetensors \
  --flux-klein-convrot-hq \
  --flux-klein-precision int8 \
  --flux-klein-policy model_quantizer_presets/flux2_klein_9b_int8_convrot_hq.json \
  --flux-klein-scale-search absmax

./venv/bin/ctq \
  -i /path/to/FLUX-2-Klein-Base-9b.safetensors \
  -o /path/to/FLUX-2-Klein-Base-9b-INT4-ConvRot-HQ.safetensors \
  --flux-klein-convrot-hq \
  --flux-klein-precision int4 \
  --flux-klein-policy model_quantizer_presets/flux2_klein_9b_int4_convrot_hq.json \
  --flux-klein-int4-layout w4a8 \
  --flux-klein-int4-group-size 16 \
  --flux-klein-scale-search absmax
```

Each compile writes a JSON sidecar report unless `--flux-klein-report` supplies another path. Confirm the report says 112 eligible projections and inspect the format counts before loading the model.

A representative native benchmark invocation is:

```bash
/path/to/ComfyUI/venv/bin/python tools/flux_klein_comfy_benchmark.py \
  --comfy-root /path/to/ComfyUI \
  --model FLUX-2-Klein-Base-9b-INT4-ConvRot-HQ.safetensors \
  --label int4_convrot \
  --output-dir /path/to/benchmark/final_images \
  --report /path/to/benchmark/runtime/int4_convrot.json \
  --steps 20 --cfg 3 --sampler euler --scheduler flux2 \
  --timed-runs 2 --resolutions 1024 1280 1536 2048
```

## Scope and next research

The selected policies are evidence-backed for FLUX.2 Klein **Base 9B**, this ComfyUI/comfy-kitchen generation, and the tested sampler family. Recalibrate rather than silently reuse the routes for distilled Klein, other parameter sizes, modified checkpoints, a different four-bit layout, or a materially different runtime kernel.

The highest-value next steps are:

1. Expand the holdout to more prompts per semantic class, especially typography and faces.
2. Evaluate multiple seeds at every resolution and report confidence intervals.
3. Re-run equal-size route swaps around the final group-16 policy; sensitivity changes after every accepted swap.
4. Test group 8 as an intermediate scale-overhead point with the final interaction-aware route seed.
5. Add activation calibration if ComfyUI exposes a stable calibrated W4 runtime layout.
6. Repeat on another Blackwell GPU and at least one pre-Blackwell target before making broad speed claims.

## Primary references

- [Black Forest Labs FLUX.2 source](https://github.com/black-forest-labs/flux2)
- [FLUX.2 Klein Base 9B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)
- [Official FLUX.2 Klein Base 9B NVFP4 model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-nvfp4)
- [ComfyUI native quantization operators](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/quant_ops.py)
- [comfy-kitchen layouts and kernels](https://github.com/Comfy-Org/comfy-kitchen)
- [Comfy model tools INT8 ConvRot reference](https://github.com/Comfy-Org/comfy-model-tools/blob/main/quant_int8_convrot.py)
- [ConvRot paper](https://arxiv.org/abs/2512.03673) and [reference implementation](https://github.com/feice-huang/ConvRot)
- [QuaRot](https://arxiv.org/abs/2404.00456)
- [SpinQuant](https://arxiv.org/abs/2405.16406)
