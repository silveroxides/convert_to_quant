# Task 3: Metadata Tensor Generation

**Objective:** Implement a metadata packing mechanism that converts quantization settings into a JSON-encoded `uint8` tensor and integrates it into the transformation pipeline. This ensures the quantized model is self-describing.

## Target Repository
All new code should be placed within the `convert_to_quant_target` directory.

## Specific Requirements

1.  **Implement Metadata Packing**:
    - Create a new file `convert_to_quant_target/convert_to_quant/converters/metadata.py`.
    - Implement a function `pack_metadata(settings: Dict[str, Any]) -> torch.Tensor`.
    - **Logic**:
        - Convert the settings dictionary to a JSON string.
        - Encode the string to bytes (utf-8).
        - Convert the bytes to a `torch.uint8` (ByteTensor) 1D tensor.
        - This mirrors the `.comfy_quant` or `.hqq_meta` pattern used in other tools.

2.  **Standardize Metadata Keys**:
    - Define a standard set of keys that should be present in the metadata.
    - Examples: `weights_dtype`, `quantized_matmul_dtype`, `block_size` (group_size), `use_svd`, `svd_rank`, etc.
    - Ensure these match the arguments used in `sdnq_math.py`.

3.  **Integrate with Transformation Pipeline**:
    - Modify `convert_to_quant_target/convert_to_quant/converters/sdnq_transform.py`.
    - Import `pack_metadata`.
    - Inside `convert_state_dict`, after successfully quantizing a layer:
        - Generate the metadata tensor for that layer using `pack_metadata(layer_settings)`.
        - Store it in `new_state_dict` with a consistent suffix.
        - **Suffix Convention**: Use `.weight_metadata` or simply `.metadata` appended to the layer name? 
            - *Recommendation*: Use `.metadata` if possible, or `.weight.metadata` if following ComfyUI patterns (often hidden/ignored). Let's stick to a clear suffix like `.metadata` or `.quant_map` for now, or follow the `.comfy_quant` convention if we want ComfyUI compatibility (which usually embeds it in a specific way, but a separate tensor is safer for `safetensors`).
            - Let's use `_metadata` suffix to match `_scale`, `_zp`. Example: `model.layers.0.linear.weight_metadata`.

4.  **Verification**:
    - Verify that the generated tensor can be decoded back to the original dictionary.

## Expected Deliverable
1.  `convert_to_quant_target/convert_to_quant/converters/metadata.py`
2.  Updated `convert_to_quant_target/convert_to_quant/converters/sdnq_transform.py`

## Preparation for Next Task
Once complete, summarize:
1.  The structure of the metadata tensor.
2.  The list of standardized keys.
3.  Pass the torch to the **Debug** agent for Phase 4 (Integration & Verification).
