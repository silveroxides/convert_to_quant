# SDNQ Refactoring Implementation Plan

This plan outlines the incremental refactoring of the `sdnq` quantization methods into a tensor-centric, stateless architecture similar to the `convert_to_quant` tool.

## Overview
The goal is to remove the dependency on `nn.Module` classes and `HfQuantizer` / `DiffusersQuantizer` frameworks during the conversion process, instead operating directly on raw tensors and state dicts.

## Phase 1: Core Math Extraction & Stateless Functions
**Agent:** 🤖 Google GenAI Developer
**Goal:** Extract the mathematical core of SDNQ into pure functions.

- Identify and pull out `quantize_weight` and `sdnq_quantize_layer_weight` math.
- Remove all `nn.Module` or `SDNQConfig` class dependencies from these functions.
- Ensure these functions accept standard Python types and `torch.Tensor` only.
- Implement standalone SVD logic if needed.

## Phase 2: State-Dict Transformation Logic
**Agent:** 💻 Code
**Goal:** Implement a dictionary-based processing pipeline.

- Create a `convert_state_dict` function that iterates over a dictionary of tensors.
- Implement logic to handle layer naming (e.g., adding `.weight`, `.weight_scale` suffixes).
- Implement exclusion logic using simple regex lists or dictionaries instead of the current class-based config.
- Ensure the output is a standard `state_dict` ready for `safetensors` saving.

## Phase 3: Metadata Tensor Generation
**Agent:** 🏗️ Architect
**Goal:** Implement metadata-driven layout descriptions.

- Implement a metadata packing function that converts quantization settings into a JSON-encoded `uint8` tensor (mirroring the `.comfy_quant` pattern).
- Integrate this into the Phase 2 pipeline so every quantized layer includes its own descriptive metadata tensor.
- Standardize the keys used in this metadata to ensure cross-tool compatibility.

## Phase 4: Integration & Verification
**Agent:** 🪲 Debug
**Goal:** Finalize the tool and verify against the target repository.

- Create a CLI entry point that uses the new stateless logic.
- Verify the generated `safetensors` files against the expectations of the `convert_to_quant` reference.
- Clean up any remaining legacy class dependencies in the refactored code.

---
*Note: Each phase must be summarized by the completing agent for the next agent.*
