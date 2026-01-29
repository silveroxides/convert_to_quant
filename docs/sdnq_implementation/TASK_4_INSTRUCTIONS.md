# Task 4: Integration & Verification

**Objective:** Finalize the refactoring by creating a CLI entry point that utilizes the new stateless pipeline (Phase 1-3) and verify that the generated models match the expected format and quality of the reference implementation.

## Target Repository
All code is in `convert_to_quant_target`.

## Specific Requirements

### 1. Create CLI Entry Point
- Create `convert_to_quant_target/convert_to_quant/cli/run_sdnq.py` (or modify `main.py`).
- It should:
    - Parse command line arguments (source model, destination, quantization config).
    - Load the model (using `safetensors` or `torch.load`).
    - **Use the new pipeline**:
        - `convert_to_quant.converters.sdnq_transform.convert_state_dict`.
    - Save the result using `safetensors`.
    - Ensure metadata is preserved/saved.

### 2. Verification
- Create a verification script `convert_to_quant_target/tests/verify_sdnq_cli.py`.
- **Goal**: Ensure the output of the new pipeline is valid and usable.
- Steps:
    - Run the new CLI on a small test model (or dummy state dict).
    - Load the resulting `safetensors` file.
    - Check for presence of quantized weights, scales, zero points, and **metadata tensors**.
    - **Compare with Reference**: If possible, compare the output structure with `convert_to_quant-reference` (though exact numerical match might differ due to stochastic rounding or library versions, structure must match).

### 3. Clean Up
- Identify any "legacy" files in `convert_to_quant_target` that are no longer needed (e.g., if there were old class-based converters copied over that are now replaced by `sdnq_transform`).
- Ensure `sdnq_math.py` and `sdnq_transform.py` do not import any unused legacy modules.

## Expected Deliverable
1.  `convert_to_quant_target/convert_to_quant/cli/run_sdnq.py` (functional CLI).
2.  `convert_to_quant_target/tests/verify_sdnq_cli.py` (passing test).
3.  A final report confirming that the refactored tool works end-to-end.

## Note for the Agent
- You are the **Debug** agent.
- Use the `execute_command` tool to run your verification scripts.
- If you find bugs in Phase 1-3 code, fix them.
