# Task 2: State-Dict Transformation Logic

**Objective:** Implement a dictionary-based processing pipeline that iterates over a model's `state_dict` and applies the stateless quantization functions from Phase 1.

## Target Repository
All new code should be placed within the `convert_to_quant_target` directory.

## Specific Requirements
1.  **Implement `convert_state_dict`**:
    - Create a new file `convert_to_quant_target/convert_to_quant/converters/sdnq_transform.py`.
    - This function should take a `state_dict` (Dict[str, Tensor]) and a configuration dictionary.
    - It should iterate over the `state_dict`, identify quantizable layers (e.g., those ending in `.weight` and matching `linear_types` or `conv_types`), and call `sdnq_quantize_layer_weight` from `sdnq_math.py`.
    - Handle layer naming: for a layer named `model.layers.0.self_attn.q_proj.weight`, the output should include:
        - `model.layers.0.self_attn.q_proj.weight` (the quantized tensor)
        - `model.layers.0.self_attn.q_proj.weight_scale`
        - `model.layers.0.self_attn.q_proj.weight_zp` (if applicable)
        - `model.layers.0.self_attn.q_proj.svd_up` (if applicable)
        - `model.layers.0.self_attn.q_proj.svd_down` (if applicable)

2.  **Exclusion and Precision Logic**:
    - Implement logic to skip certain layers based on regex patterns or exact names (similar to `modules_to_not_convert`).
    - Implement logic to use different dtypes for specific layers (similar to `modules_dtype_dict`).
    - Use a simple dictionary for these configurations instead of the legacy `SDNQConfig` class.

3.  **Integration with Phase 1**:
    - Import `sdnq_quantize_layer_weight` from `convert_to_quant.converters.sdnq_math`.
    - Import `linear_types`, `conv_types`, `conv_transpose_types` from `convert_to_quant.converters.constants`.

4.  **Output**:
    - The function should return a new `state_dict` containing both the original unquantized tensors (that were skipped) and the new quantized tensors + their auxiliary tensors (scales, ZPs, etc.).

## Expected Deliverable
A file `convert_to_quant_target/convert_to_quant/converters/sdnq_transform.py` containing the `convert_state_dict` logic.

## Preparation for Next Task
Once complete, summarize:
1.  How the state dict is iterated and transformed.
2.  How naming suffixes are handled.
3.  Review Phase 3 of `REFACTOR_PLAN.md`.
