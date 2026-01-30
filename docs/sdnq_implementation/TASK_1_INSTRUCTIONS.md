# Task 1: Stateless Math Extraction

**Objective:** Extract the mathematical quantization logic from the `sdnq` implementation in `quant_extract/rel_path_sweep/modules/sdnq/quantizer.py` and refactor it into pure, stateless functions.

## Target Repository
All new code should be placed within the `convert_to_quant_target` directory.

## Specific Requirements
1.  **Isolate `quantize_weight`**:
    - Relocate the logic from [`quantize_weight`](quant_extract/rel_path_sweep/modules/sdnq/quantizer.py:46) to a new file `convert_to_quant_target/convert_to_quant/converters/sdnq_math.py`.
    - Ensure it takes parameters like `weight` (Tensor), `reduction_axes`, `weights_dtype` (string/enum), and `use_stochastic_rounding` (bool).
    - It must NOT depend on `devices.inference_context()` or any global state from the `sdnext` repository.

2.  **Isolate `sdnq_quantize_layer_weight`**:
    - Extract the logic from [`sdnq_quantize_layer_weight`](quant_extract/rel_path_sweep/modules/sdnq/quantizer.py:208).
    - Refactor it to accept a `torch.Tensor` and a dictionary of settings (instead of an `SDNQConfig` object).
    - Return a tuple of raw tensors: `(quantized_weight, scale, zero_point, svd_up, svd_down)`.
    - Remove the instantiation of `SDNQDequantizer`. We will handle metadata separately in Phase 3.

3.  **Standalone SVD Logic**:
    - Extract [`apply_svdquant`](quant_extract/rel_path_sweep/modules/sdnq/quantizer.py:71) and ensure it is a pure function.

4.  **Dependencies**:
    - Use `torch` and `math` only.
    - If constants from [`common.py`](quant_extract/rel_path_sweep/modules/sdnq/common.py) are needed (like `dtype_dict`), copy them into a local `constants.py` or similar in the target directory to ensure the new code is self-contained.

## Expected Deliverable
A file [`convert_to_quant_target/convert_to_quant/converters/sdnq_math.py`](convert_to_quant_target/convert_to_quant/converters/sdnq_math.py) containing the extracted and refactored functions.

## Preparation for Next Task
Once complete, you must:
1.  Summarize which functions were extracted and any changes made to the parameter signatures to achieve statelessness.
2.  Review Phase 2 of [REFACTOR_PLAN.md](../REFACTOR_PLAN.md).
3.  Write a new file `TASK_2_INSTRUCTIONS.md` for the next agent (Code Mode), detailing how to implement the `convert_state_dict` logic using the stateless functions you just created.
