# Local INT8 ConvRot Group-64 Pipeline Correction

## Objective

Update both primary local INT8 row-wise ConvRot flows—simple and
learned/AdaRound—so layers that cannot use group 256 can use valid group-64
regular ConvRot, while preserving the existing local conversion pipeline and
output contract.

Comfy-kitchen version `0.2.31`, commit
`7c6ca3a5b63857d42c2d49777d6afb69de23f13f`, is a pinned, non-authoritative
mathematical comparison source. Its source contains the
regular-Hadamard construction, weight/activation rotation equations, and
row-wise INT8 scale/rounding equations against which the local mathematics must
be compared. It has no authority over this repository's architecture,
dependencies, control flow, state ownership, calibration, bias correction, or
serialization. Current ComfyUI source separately defines the runtime metadata
contract that this repository's output must satisfy.

The re-analysis must inspect current comfy-kitchen only to compare equations.
Neither agreement nor divergence authorizes a local code change. Any proposed
local change must first be justified from the clean local execution path and a
locked decision in this file. A mathematical divergence is recorded and
requires user direction.

## Restart Authority and Mandatory Re-analysis

This file is the complete restart authority and implementation plan for a new session. A future agent must assume
that no conversational context survives and that the current working-tree
implementation draft is invalid. The inspection findings are leads to verify,
not permission to skip re-analysis.

No implementation may begin until the re-analysis gate below is completed and
recorded. The re-analysis target is the repository's own pipeline and output
contract. Comfy-kitchen supplies only mathematical equations for comparison.
Current ComfyUI supplies only the runtime loading and metadata interpretation
contract.

### Authority hierarchy

Apply this order without exception:

1. Explicit user decisions locked in this file.
2. Clean local `HEAD` code for execution order, ownership, state lifecycle,
   public interfaces, existing metadata fields, and behavior outside the
   target defect.
3. Current ComfyUI source only for documenting how emitted tensors and metadata
   are consumed at runtime; it cannot redesign the converter.
4. Comfy-kitchen source only for non-authoritative mathematical comparison; it
   cannot override local state, control flow, APIs, metadata, or dependencies.

When sources conflict, do not choose a winner by convenience. Preserve clean
local behavior and record the conflict for the user unless this file contains
an explicit locked decision resolving it.

### Required initialization order

1. Discover applicable `AGENTS.md` files by path/existence metadata before
   reading source. Read the user rules supplied to the new session and every
   applicable repository rule using bounded structural reads.
2. Read this entire plan. Do not read historical inspection handoffs, rollout
   JSONL files, or other session-history artifacts. Re-derive the required
   evidence from clean `HEAD` and the exact sources mandated below.
3. Obtain `git status --short --branch`, then `git diff --stat` and
   `git diff --name-only`. Do not inspect a broad diff before those commands.
4. Treat `HEAD` as the source baseline for every agent-created modified file.
   Inspect baseline symbols with `git show HEAD:<path>` using symbol-bounded
   extraction. Do not use the invalid working-tree implementation to infer the
   original execution flow.
5. Do not edit, format, install, convert a model, or run a broad test suite
   during re-analysis.

### Mandatory low-overhead coordinator and immediate subagent execution

Run the primary agent at `low` reasoning. Do not use `Ultra`, `max`, or `xhigh`
for this task. After the primary reads only the required rules and this plan, it
must immediately spawn all three analysis subagents before inspecting local
source, model artifacts, or external comparison/runtime sources itself.

Run the three prescribed subagents at `medium` reasoning in parallel. Give each
subagent a self-contained, bounded prompt without a full-history fork and
require a concise evidence report. A permissive thread cap is capacity, not an
instruction to fill every slot. Spawn an additional reviewer only for a
specific evidence gap that cannot be covered by the three prescribed tracks.

The primary agent owns correctness and scope control. The user must not be
expected to monitor execution, catch mistakes, reconcile agent findings, or
repeat locked decisions. The primary agent must proactively audit every
checkpoint against the authority hierarchy, protected state, diff allowlist,
and validation gates. User input is required only when a source contradiction
needs new authority and cannot be resolved by the plan or bounded inspection.

Partition read-only work as follows. Subagents must not edit files, install
packages, inspect session history, or decide scope.

| Owner | Mandatory responsibility | Required output |
| --- | --- | --- |
| Primary agent | Rules, this plan, working-tree provenance, final reconciliation | One consolidated evidence ledger and explicit gate decision |
| Local-pipeline subagent | Trace the clean `HEAD` execution paths for primary simple and primary learned/AdaRound INT8 row-wise ConvRot | Symbol-level call/data-flow report, including every phase and state owner |
| Serialization/runtime subagent | Trace local serialization and current ComfyUI loading/runtime interpretation | Exact tensor names, metadata keys, group propagation, and failure consequences |
| Arithmetic/tests subagent | Compare local group selection/Hadamard/quant arithmetic with comfy-kitchen's mathematical equations and map test coverage | Equation-level comparison and exact missing-test list |

The primary agent must independently inspect every source hunk that affects the
final design. Subagent summaries may locate evidence but cannot replace primary
verification. Conflicting findings stop the gate until reconciled against
source.

### Mandatory local baseline inspection

Inspect these exact clean-`HEAD` symbols and no broader source unless a directly
referenced identifier requires it:

- `convert_to_quant/utils/convrot.py`
  - `build_hadamard`
  - `rotate_weight`
  - `rotate_activation`
- `convert_to_quant/converters/learned_rounding.py`
  - `LearnedRoundingConverter.__init__`
  - `LearnedRoundingConverter.convert`
  - `LearnedRoundingConverter._convert_int8_tensorwise`
- `convert_to_quant/comfy/quant_ops.py`
  - `TensorWiseINT8Layout.quantize`
  - `TensorWiseINT8Layout.dequantize`
- `convert_to_quant/utils/tensor_utils.py`
  - `prepare_calibration_data`
- `convert_to_quant/formats/fp8_conversion.py`
  - converter-kwargs construction
  - `create_converter_for_format`
  - primary/custom/fallback converter selection
  - pre-conversion ConvRot prediction
  - converter invocation/unpacking
  - INT8 tensor and `.comfy_quant` serialization
  - `_quantization_metadata` construction
  - bias-correction application
- `convert_to_quant/utils/comfy_quant.py`
  - `create_comfy_quant_tensor`
- `convert_to_quant/cli/main.py`
  - parser definitions and destination names for `--comfy_quant`,
    `--save-quant-metadata`, `--verbose`, `--low-memory`, the generated model
    filter flags, `--int8`, `--scaling_mode`/`--scaling-mode`, `--convrot`,
    `--convrot-group-size`, and `--simple`
  - the `run_conversion` call arguments that propagate those destinations into
    `convert_to_fp8_scaled`
- Tests:
  - `tests/test_int8_row_convrot.py`
  - `tests/test_convrot.py`
  - `tests/test_int8_tensorwise.py`
  - `tests/test_int8_rowwise_fix.py`
  - every test function in `tests/test_cli_consistency.py` that names any parser
    destination listed above; enumerate those function names in the evidence
    ledger before reading their complete function boundaries

The local report must explicitly establish:

1. The exact call path from parsed primary flags to converter construction.
2. How primary, custom, fallback, and layer-config converters inherit or
   override `no_learned_rounding`, ConvRot, and group size.
3. The ordered phases inside `_convert_int8_tensorwise`, including the tensors'
   coordinate domains before and after rotation.
4. Every consumer of returned `qdata`, scale, dequantized weight, and
   `extra_tensors`.
5. The all-zero shortcut and its current row-wise INT8 scale behavior.
6. Where ConvRot success is currently inferred versus actually observed.
7. Whether mutable converter state is safe across sequential layers and OOM
   retries.
8. Every currently tested behavior that the targeted change could affect.

### Mandatory model evidence

Revalidate the reference artifact without loading the full model:

- Model:
  `F:\models\diffusion_models\minimax_h3_fl2va_int8_convrot.safetensors`
- Use only `UnifiedSafetensorsLoader(..., low_memory=True)`.
- Decode individual `.comfy_quant` tensors with `tensor_to_dict`.
- Verify at least:
  - `blocks.0.adaln_proj.linear.weight` has input width 2688;
  - its `.comfy_quant` dictionary uses group 64;
  - one compatible non-2688 quantized layer uses group 256;
  - corresponding weight-scale shapes/dtypes;
  - whether file-header `_quantization_metadata` exists.
- Do not infer that the historical skip log and this model came from the same
  conversion unless provenance is independently proven.

### Mandatory mathematical-comparison and runtime-contract inspection

Use GitHub API/source inspection. Do not execute comfy-kitchen as part of the
target implementation. Record the pinned commit above, then determine the
latest published comfy-kitchen version and current upstream commit. Compare
current source with the pinned commit only to detect mathematical divergence.
Inspect only these mathematical definitions and tests at both revisions when
they differ:

- `comfy_kitchen/tensor/int8_utils.py`
  - regular-Hadamard validation and construction
  - weight and activation rotation equations
- `comfy_kitchen/backends/eager/quantization.py`
  - row-wise INT8 scale, zero-scale handling, rounding, and clamp range
- `comfy_kitchen/tensor/int8.py`
  - ConvRot group validation and the arguments passed into mathematical
    quantization; do not adopt its object model or serialization architecture
- `tests/test_int8.py`
  - group-64 mathematical tests, divisibility tests, and eager arithmetic
    expectations
- Separately inspect current ComfyUI `comfy/ops.py` as the runtime-contract
  source for `.comfy_quant` parsing, propagation of `convrot` and
  `convrot_groupsize`, and state-dict serialization of those fields.

The report must separate three categories with no authority transfer between
them:

- **Comfy-kitchen mathematical comparison:** regular-Hadamard matrix family,
  offline/online rotation equations, scale shape/dtype, zero-scale handling,
  division, rounding, and clamp equations.
- **ComfyUI runtime contract:** exact tensor names and metadata fields read at
  load time and propagated into runtime activation rotation.
- **Forbidden authority transfer:** no comfy-kitchen imports, calls, vendoring,
  dependency, object model, pipeline ownership, calibration flow, bias flow, or
  serialization architecture may be adopted.

### Re-analysis evidence ledger and implementation gate

Before any implementation, append a dated `Re-analysis Evidence Ledger` to
this file containing:

- clean baseline commit ID and branch;
- invalid working-tree files and exact agent-owned hunks to remove;
- pinned comfy-kitchen version/commit, current comfy-kitchen version/commit,
  any mathematical divergence between them, and current ComfyUI commit;
- verified local pipeline phase diagram in ordered prose;
- verified group-selection defect;
- exact local-versus-comfy-kitchen mathematical differences;
- exact serialization/runtime contract;
- tests that cover the behavior and tests that must be added;
- confirmation that every protected variable and protected section below can
  remain invariant;
- unresolved contradictions, which must be `None` before implementation.

Implementation is forbidden unless the ledger concludes all of the following:

- group 64 is already the same regular-Hadamard family locally and externally;
- the defect can be corrected inside the existing local execution flow;
- no external dependency or pipeline replacement is necessary;
- the targeted change can include primary simple and primary learned/AdaRound
  while remaining isolated from custom/fallback/FP8 paths;
- actual applied group can be serialized without changing public return types;
- unresolved contradictions are `None`.

If any conclusion fails, stop and ask the user. Do not invent a fallback design.

### Mandatory checkpoint commits and final squash

The agent must preserve each verified task-owned state in Git so recovery never
depends on user monitoring:

1. Record the baseline branch and commit before the first task mutation. Never
   stage or commit pre-existing user changes, unrelated files, or unrelated
   hunks.
2. After each coherent implementation checkpoint passes its focused validation
   and scope-diff audit, create a checkpoint commit. Cover at minimum the local
   resolver, simple-path arithmetic, learned/AdaRound applied-group state,
   serialization/documentation, and final audit corrections as separate
   recoverable states when they change tracked content.
3. Give the first checkpoint the intended concise final feature message. Make
   later checkpoint commits `fixup!` commits targeting it. Never amend, reorder,
   or squash any pre-existing or user-authored commit.
4. After all acceptance criteria pass and unresolved contradictions are exactly
   `None`, record the completed tree ID and create a temporary backup ref at the
   checkpoint tip. Autosquash only the contiguous task-owned checkpoint series.
5. Verify that the squashed commit has the identical tree ID, rerun the final
   focused validation, and audit its diff against the recorded baseline. Remove
   the temporary backup ref only after those checks pass.

The completed branch must contain one concise full-feature implementation
commit for this task, with no loss of any verified checkpoint content and no
unrelated changes.

## Locked Decisions and Protected State

The following values are user-approved and immutable for this task. They are
not defaults to reconsider and not assumptions to optimize.

### Protected variables

| Variable/contract | Locked value |
| --- | --- |
| Target CLI paths | Primary `--int8 --scaling-mode row --convrot`, with or without `--simple`; the existing `--scaling_mode row` alias is equivalent |
| Implementation owner | This repository's existing local pipeline |
| External dependency | None; comfy-kitchen supplies mathematical equations only |
| Preferred default group | 256 |
| Automatic fallback | 64 only, and only when the parsed CLI value of `--convrot-group-size` is 256 and 256 does not divide the width |
| Explicit non-256 group | Exact-only; do not silently substitute another size |
| Group 64 matrix | Existing local regular-Hadamard construction |
| Group 256 matrix | Existing local regular-Hadamard construction |
| Simple-mode initial row quantization | Use the specified comparison equations in section 5 |
| Learned/AdaRound quantization and optimization | Preserve clean-HEAD mathematics; only group resolution and applied-group metadata change |
| Learned/AdaRound all-zero shortcut | Preserve clean-HEAD behavior unchanged |
| Custom/fallback simple modes | Unchanged |
| Public converter return tuple | Unchanged |
| Bias-correction owner and sign | Existing pipeline, unchanged |
| `full_precision_matrix_mult` | Preserve exactly when requested/configured |
| `per_row` metadata | Preserve clean-HEAD behavior unchanged |
| `orig_dtype` metadata | Preserve clean-HEAD behavior unchanged |
| FP8 behavior and metadata | Unchanged |
| Broad dependency/package changes | Forbidden |

### Locked baseline invocation profile

The representative learned/AdaRound conversion is:

```powershell
ctq -i <INPUT.safetensors> -o <OUTPUT.safetensors> --comfy_quant --save-quant-metadata --verbose VERBOSE --low-memory --minimaxh3 --int8 --scaling_mode row --convrot --convrot-group-size 256
```

The representative simple conversion is the identical command with `--simple`
added. `<INPUT.safetensors>` and `<OUTPUT.safetensors>` are caller-supplied paths
and are not hard-coded by this task.

`--minimaxh3` is a representative existing model-filter flag. It is not a
Minimax-specific scope decision. Any one existing filter flag may occupy that
slot in a real invocation. The implementation must not inspect filter identity
when resolving ConvRot groups, quantizing rows, recording applied groups, or
serializing metadata.

For both commands, preserve these meanings exactly:

| Argument | Required preserved effect |
| --- | --- |
| `--comfy_quant` | Write per-layer `.comfy_quant` tensors using the existing local serialization path |
| `--save-quant-metadata` | Write the existing `_quantization_metadata` header structure with the actual applied ConvRot group |
| `--verbose VERBOSE` | Preserve current logging selection; it must not alter quantization decisions |
| `--low-memory` | Preserve streaming input behavior and tensor-at-a-time conversion ownership |
| `--minimaxh3` or another existing filter flag | Preserve the selected filter's current layer inclusion/exclusion behavior; filter identity must not affect the new ConvRot logic |
| `--int8` | Preserve INT8 converter selection |
| `--scaling_mode row` | Preserve row-wise INT8 layout and scale shape |
| `--convrot` | Preserve local offline rotation and runtime metadata declaration |
| `--convrot-group-size 256` | Request 256; for the target primary paths only, fall back to 64 when 256 cannot divide the input width |
| `--simple` when present | Disable learned/AdaRound optimization only; it does not select a different pipeline owner |

Re-analysis must trace every argument in this table from parser destination to
the exact branch or serialized output it controls. It must also verify that all
filter flags converge on shared conversion behavior before the new ConvRot
hook. Integration tests must call the Python conversion entry point with
equivalent values, use a synthetic filter-compatible tensor name, and must not
invoke a subprocess CLI or perform a full model conversion.

### Protected code sections

These sections may receive the minimum conditional hook described later, but
must not be moved, duplicated, deleted, reordered, or assigned new ownership:

- `LearnedRoundingConverter.convert` retry/OOM lifecycle.
- `_convert_int8_tensorwise` ConvRot rotation block.
- `_convert_int8_tensorwise` Phase 2 calibration block.
- `_convert_int8_tensorwise` initial quantization block.
- `_convert_int8_tensorwise` common dequantization statement.
- `_convert_int8_tensorwise` Phase 4 residual bias calibration block.
- `_convert_int8_tensorwise` cleanup and return.
- Outer conversion bias-correction application.
- `full_precision_matrix_mult` precedence and serialization.
- FP8 serialization branches.

### Forbidden assumptions

- Do not assume simple, custom-simple, and fallback-simple are interchangeable.
- Do not assume metadata divisibility proves rotation succeeded.
- Do not assume an all-zero weight may bypass the targeted scale contract.
- Do not infer architecture, control flow, state ownership, or serialization
  design from comfy-kitchen's mathematical implementation.
- Do not assume extra metadata keys are harmless to remove or add.
- Do not assume the inspected model was generated by the historical log.
- Do not infer user approval from implementation convenience, performance, or
  external kernel availability.

## Non-Negotiable Scope Boundaries

- Do not add, import, call, install, or vendor comfy-kitchen.
- Do not move logic out of `LearnedRoundingConverter._convert_int8_tensorwise`.
- Do not add an early return or replacement conversion branch.
- Preserve the existing sequence: local group resolution, local weight
  rotation, calibration preparation, initial quantization, common
  dequantization, residual bias calibration, cleanup, and return.
- Do not change FP8, tensor-wise INT8, block-wise INT8, custom/fallback
  converters, or layer-config-owned behavior.
- `--dynamic-convrot` is outside scope: leave its code and tests unchanged and
  do not otherwise discuss, redesign, or add coverage for it in this task.
- Do not change `full_precision_matrix_mult` handling.
- Do not change the existing Hadamard construction in this task. Groups 64 and
  256 already use the local regular-Hadamard `H4` Kronecker construction. The
  existing Sylvester fallback for other sizes is outside this task.
- Do not change the public converter return tuple or existing calibration/bias
  correction ownership.

## Established Technical Facts

| Concern | Established behavior |
| --- | --- |
| Local group 64 | Already uses regular Hadamard because `64 = 4^3` |
| Local group 256 | Already uses regular Hadamard because `256 = 4^4` |
| Current failure | Static group 256 is skipped for width 2688 even though regular group 64 divides that width |
| Required selection | Prefer 256 where divisible; otherwise use 64 where divisible |
| Offline transform | Split `[out,in]` weights into groups and calculate `W_rot = W @ H_block.T` |
| Runtime transform | ComfyUI reads `convrot` and `convrot_groupsize` and applies the matching activation rotation |
| Group-64 runtime | Valid regular ConvRot; it does not require a specialized fused group-256 kernel |
| Reference metadata | The inspected 2688-wide model uses `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":64}` |

## Exact Implementation

### 1. Remove the invalid draft before new implementation

- The invalid draft began from a tracked-clean working tree; the initial
  untracked files were inspection and session-history artifacts. Outside this
  context window, restore exactly these tracked paths from `HEAD`:

  ```powershell
  git restore -- pyproject.toml requirements.txt convert_to_quant/utils/convrot.py convert_to_quant/converters/learned_rounding.py convert_to_quant/formats/fp8_conversion.py
  ```

- Do not run `git clean`, `git reset`, `git checkout .`, or restore any path not
  listed in that command.
- Preserve all pre-existing user changes and untracked inspection/session files.
- Keep this corrected plan file.
- Confirm the working diff no longer contains `comfy-kitchen`,
  `use_comfy_kitchen_convrot`, the invalid early return, or relocated
  original-domain bias correction.

### State-preservation rule

After invalid-draft cleanup, clean local `HEAD` behavior is frozen by default.
An implementation diff is permitted only for the exact symbols and conditional
hooks named in sections 2 through 6. Every other line and every behavior not
explicitly changed there must remain unchanged.

The final diff allowlist is:

- one new group-resolution function in `utils/convrot.py`;
- one internal constructor flag, one internal applied-group field, their reset
  assignments, and conditional use in `LearnedRoundingConverter`;
- one module-local arithmetic helper and one call to it inside the existing
  initial row-wise quantization branch;
- one primary-converter flag assignment plus one targeted post-conversion
  metadata override in `formats/fp8_conversion.py`;
- focused tests and the INT8 ConvRot README text;
- this plan and its appended evidence ledger.

Any diff outside this allowlist is a gate failure and must be removed, not
explained as cleanup, refactoring, consistency, optimization, or convenience.

### 2. Add a local primary-ConvRot group resolver

- Add `resolve_int8_convrot_group_size(in_features, requested_group_size)` in
  `convert_to_quant/utils/convrot.py`.
- For requested group 256, return 256 when divisible; otherwise return 64 when
  divisible; otherwise return `None`.
- For any explicitly requested group other than 256, preserve clean-HEAD
  exact-group
  behavior: return it only when divisible, otherwise return `None`.
- Required assertions:
  - `(512, 256) -> 256`
  - `(4096, 256) -> 256`
  - `(2688, 256) -> 64`
  - `(1152, 256) -> 64`
  - `(96, 256) -> None`
  - `(1024, 64) -> 64`
  - `(768, 1024) -> None`

### 3. Mark only primary INT8 converters

- Add `primary_int8_convrot_compat: bool = False` to
  `LearnedRoundingConverter.__init__`. It permits primary 256-to-64 resolution
  and applied-group status for both simple and learned flows. It permits the
  section-5 arithmetic helper only when `self.no_learned_rounding` is true.
- Inside `create_converter_for_format`, set it to
  `bool(is_primary and fmt == "int8")`
  before constructing `LearnedRoundingConverter`.
- Set it to false unconditionally for non-primary converters before applying
  custom or fallback overrides. No override dictionary may set it. Therefore
  custom and fallback converters retain their current behavior, whether simple
  or learned.

### 4. Divert locally inside the existing ConvRot flow

- In the existing static/exact-group branch of the "Apply ConvRot" block, use
  the new resolver only when `self.primary_int8_convrot_compat` is true.
  Custom and fallback modes retain their current exact group.
- Set a local `target_convrot_applied = False` before group resolution. Set it
  true only when the new resolver selected the group and local rotation
  succeeded. This boolean, not a CLI option, gates the arithmetic and status
  changes below.
- Continue using the existing local `build_hadamard` and `rotate_weight` calls.
  Do not replace them and do not change their exception behavior in this task.
- Record the resolved group as successfully applied only after local rotation
  returns successfully.
- Allow targeted primary simple ConvRot all-zero weights to enter
  `_convert_int8_tensorwise` instead of taking the generic all-zero shortcut;
  this preserves the same pipeline and produces a valid row scale.
- Preserve the clean-HEAD all-zero shortcut for learned/AdaRound conversion.

### 5. Match the specified row-wise INT8 equations locally

- Add module-local helper
  `_quantize_convrot_int8_rowwise(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`
  in `convert_to_quant/converters/learned_rounding.py`.
- Call it only at the existing initial row-wise quantization decision when
  `target_convrot_applied and self.no_learned_rounding` is true.
- Input is the already locally rotated `W_float32`; no rotation occurs in the
  helper.
- Compute exactly:
  1. `abs_max = W_float32.abs().amax(dim=-1, keepdim=True)`
  2. `scale = (abs_max.float() / 127.0).clamp(min=1e-30)`
  3. Convert `scale` to the weight device/dtype for division, replacing exact
     zeros with `torch.finfo(W_float32.dtype).tiny` if necessary.
  4. `qdata = (W_float32 / scale_for_math).round().clamp(-128.0, 127.0).to(torch.int8)`
- Return `qdata` and FP32 `scale` with shape `[out_features,1]`.
- Continue through the existing common local dequantization, Phase 4 residual
  bias calibration, cleanup, and return. Do not duplicate or relocate them.
- Learned/AdaRound ConvRot, non-ConvRot row-wise INT8, and custom/fallback
  ConvRot continue using clean-HEAD `TensorWiseINT8Layout.quantize` and optimizer
  behavior.

### 6. Serialize confirmed local execution

- Add `last_primary_convrot_group_size: Optional[int] = None` to the converter.
  Reset it at the start of `convert` and at the start of every retry. Set it to
  `layer_group_size` only after successful local rotation and only when
  `target_convrot_applied` is true.
- In `convert_to_fp8_scaled`, for targeted primary ConvRot paths only, read this
  post-conversion state instead of predicting success before conversion.
- Leave existing metadata resolution unchanged for custom, fallback,
  layer-config-owned, and non-target formats.
- For successfully applied targeted primary ConvRot:
  - store INT8 weight data unchanged;
  - store `.weight_scale` as detached CPU FP32 `[out_features,1]`;
  - write flat `.comfy_quant` fields `format="int8_tensorwise"`,
    `convrot=true`, and `convrot_groupsize=<actual group>`;
  - preserve `full_precision_matrix_mult=true` when explicitly enabled;
  - preserve baseline `per_row` and `orig_dtype` behavior unchanged;
  - mirror the actual group in `_quantization_metadata` when requested.
- If no group is compatible or local rotation fails, retain ordinary local
  row-wise quantization and omit ConvRot fields for the targeted primary path.

### 7. Documentation

- Update only the README INT8 ConvRot section. Show the existing learned command
  and its `--simple` variant with `--int8 --scaling-mode row --convrot`.
- State that default group 256 remains when compatible and falls back to
  regular group 64 for widths such as 2688.
- Do not change FP8 documentation or unrelated CLI behavior.

## Focused Test Plan

- Resolver tests cover every required assertion. They must also prove an
  explicit non-256 group retains exact behavior; use `(1024, 512) -> 512` and
  `(768, 512) -> None` without changing `build_hadamard`.
- Local arithmetic tests compare the targeted helper with independently
  calculated equation results for random rows, zero rows, group 64, and group
  256. Assert exact qdata, FP32 scale, and `[rows,1]` shape.
- Pipeline tests run both primary simple and primary learned/AdaRound flows and
  prove width 512 uses 256, width 2688 uses 64, width 96 skips ConvRot, common
  local dequantization remains in use, and existing Phase 4 bias correction
  remains populated when bias exists.
- The learned test must prove group 64 reaches the existing optimizer while the
  section-5 simple arithmetic helper is not called. It must not reduce iteration
  count to zero or set `no_learned_rounding=True`.
- Scope-regression tests prove custom simple/learned and fallback
  simple/learned do not use the new resolver/arithmetic.
- Synthetic safetensors tests assert actual groups 256/64, absent ConvRot fields
  for incompatibility, preserved `full_precision_matrix_mult`, FP32 row scales,
  and optional header agreement.
- Run only focused ConvRot, targeted metadata, and directly affected
  CLI-consistency tests. Do not run the full repository suite.

## Acceptance Criteria

- No comfy-kitchen dependency, import, runtime call, or vendored code exists.
- Primary simple and primary learned/AdaRound INT8 row-wise ConvRot pipelines
  have no new early return and retain their original phase order and ownership.
- A 2688-wide layer converts with regular group-64 ConvRot and truthful metadata.
- Compatible layers continue using group 256.
- Non-target conversion modes have no behavioral diff.
- Explicit full-precision matrix multiplication remains preserved.
- The user is not required to supervise analysis, implementation, scope
  enforcement, subagent reconciliation, or validation.
- Verified checkpoint commits are autosquashed into one concise task-owned
  feature commit whose tree matches the validated pre-squash tree exactly.

## Re-analysis Evidence Ledger Template

Do not delete this template. Before implementation, copy it immediately below
this section, replace every `<REQUIRED>` value, and set the gate to `PASS` or
`FAIL`. A value may not be left blank and may not use uncertain language such
as “likely,” “probably,” “appears,” “should,” or “assume.” Source paths plus
symbol names or commit IDs are required.

### Ledger identity

| Field | Required value |
| --- | --- |
| Re-analysis date/timezone | `<REQUIRED>` |
| Clean baseline branch | `<REQUIRED>` |
| Clean baseline commit | `<REQUIRED>` |
| Invalid tracked draft paths | `<REQUIRED>` |
| Pinned mathematical-comparison commit | `7c6ca3a5b63857d42c2d49777d6afb69de23f13f` |
| Current comfy-kitchen commit/version inspected | `<REQUIRED>` |
| Mathematical divergence from pinned commit | `<REQUIRED: exact differences or None>` |
| Current ComfyUI commit inspected | `<REQUIRED>` |

### Clean local pipeline trace

Fill every row with the clean-HEAD symbol and the exact input/output state.

| Order | Phase | Clean-HEAD symbol | Input state | Output state/owner |
| ---: | --- | --- | --- | --- |
| 1 | CLI destination propagation | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| 2 | Primary converter construction | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| 3 | All-zero decision | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| 4 | Static ConvRot group decision | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| 5 | Local regular-Hadamard weight rotation | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| 6 | Phase 2 calibration preparation | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| 7 | Initial row-wise quantization | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| 8 | Learned/AdaRound optimization or simple bypass | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| 9 | Common dequantization and Phase 4 bias correction | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| 10 | Tensor/metadata serialization and outer bias application | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |

### Mathematical comparison

| Operation | Clean local equation | Pinned comfy-kitchen equation | Required local change |
| --- | --- | --- | --- |
| Group-64 matrix construction | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED: exact change or None>` |
| Group-256 matrix construction | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED: exact change or None>` |
| Weight rotation | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED: exact change or None>` |
| Activation rotation | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED: exact change or None>` |
| Row absmax and stored scale | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED: exact change or None>` |
| Scale used for division | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED: exact change or None>` |
| Rounding and clamp | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED: exact change or None>` |

### Runtime and artifact contract

| Evidence | Required verified value |
| --- | --- |
| 2688-wide model tensor key and shape | `<REQUIRED>` |
| Its decoded `.comfy_quant` dictionary | `<REQUIRED>` |
| Deterministically selected group-256 comparison key and shape | `<REQUIRED>` |
| Its decoded `.comfy_quant` dictionary | `<REQUIRED>` |
| Both `.weight_scale` shapes/dtypes | `<REQUIRED>` |
| File-header metadata keys | `<REQUIRED>` |
| Current ComfyUI load symbols/fields | `<REQUIRED>` |
| Current ComfyUI runtime group propagation | `<REQUIRED>` |

### Scope-preservation proof

| Protected area | Evidence it remains unchanged |
| --- | --- |
| FP8 | `<REQUIRED>` |
| Tensor-wise/block-wise INT8 | `<REQUIRED>` |
| Custom/fallback converters | `<REQUIRED>` |
| Layer-config-owned behavior | `<REQUIRED>` |
| Learned/AdaRound optimizer mathematics | `<REQUIRED>` |
| Calibration and bias correction | `<REQUIRED>` |
| Public return tuple | `<REQUIRED>` |
| `full_precision_matrix_mult` | `<REQUIRED>` |
| `per_row` and `orig_dtype` metadata | `<REQUIRED>` |
| Filter selection | `<REQUIRED>` |
| Low-memory loading | `<REQUIRED>` |

### Test delta and contradictions

| Field | Required value |
| --- | --- |
| Existing tests proving preserved behavior | `<REQUIRED: exact test names>` |
| Missing tests to add | `<REQUIRED: exact proposed test names and assertions>` |
| Unresolved contradictions | `<REQUIRED: None or exact contradiction>` |
| Re-analysis gate | `<REQUIRED: PASS or FAIL>` |

`PASS` is permitted only when `Unresolved contradictions` is exactly `None`
and every earlier field is populated with source-backed evidence. `FAIL` ends
the task before implementation and requires user direction.

## Re-analysis Evidence Ledger — 2026-08-30

### Ledger identity

| Field | Verified value |
| --- | --- |
| Re-analysis date/timezone | `2026-08-30 16:50:51 +02:00`, Europe/Stockholm |
| Clean baseline branch | `feature/minimaxh3-filter` |
| Clean baseline commit | `1e401d4274c5318f90537b21fa93e7a99d3e044f` |
| Invalid tracked draft paths | None. `git status --short --branch`, `git diff --stat`, and `git diff --name-only` proved the tracked tree matched `HEAD`; the restart/plan/session artifacts were untracked and preserved. |
| Pinned mathematical-comparison commit | comfy-kitchen `0.2.31`, `7c6ca3a5b63857d42c2d49777d6afb69de23f13f` |
| Current comfy-kitchen commit/version inspected | `main` at `dae00a13d458876570804523ae045a487fd92961`; latest published tag remains `v0.2.31` at the pinned commit |
| Mathematical divergence from pinned commit | None in `comfy_kitchen/tensor/int8_utils.py`, `comfy_kitchen/backends/eager/quantization.py`, `comfy_kitchen/tensor/int8.py`, or the focused `tests/test_int8.py` mathematics. |
| Current ComfyUI commit inspected | `Comfy-Org/ComfyUI` `master` at `8a33128f2f8c5585c57486c07de481241e70a39c` |

### Clean local pipeline trace

| Order | Phase | Clean-HEAD symbol | Input state | Output state/owner |
| ---: | --- | --- | --- | --- |
| 1 | CLI destination propagation | `convert_to_quant/cli/main.py:get_parser`, `extract_filter_flags`, `run_conversion` | Parsed `comfy_quant`, `save_quant_metadata`, `verbose`, `low_memory`, generated filter flags, `int8`, `scaling_mode`, `convrot`, `convrot_group_size`, and `simple` destinations | `run_conversion` configures verbosity, resolves filters, and passes the other exact values to `formats.fp8_conversion.convert_to_fp8_scaled`; the configured group means the parsed `args.convrot_group_size`. |
| 2 | Primary converter construction | `convert_to_quant/formats/fp8_conversion.py:convert_to_fp8_scaled.create_converter_for_format` | Primary format plus copied global converter kwargs | Primary `LearnedRoundingConverter` inherits global simple/ConvRot/group settings; non-primary custom/fallback reset simple before their explicit overrides; layer-config selection remains separately owned. |
| 3 | All-zero decision | `convert_to_quant/converters/learned_rounding.py:LearnedRoundingConverter.convert` | GPU/compute-dtype `W_float32` | Clean `HEAD` returns zero qdata/dequantized data and an INT8 block-grid ones scale before row mode is considered; the targeted primary-simple ConvRot path must bypass only this shortcut, while learned/AdaRound behavior remains unchanged. |
| 4 | Static ConvRot group decision | `LearnedRoundingConverter._convert_int8_tensorwise` | Row-wise INT8 converter fields and weight width `N` | Clean `HEAD` accepts only the configured exact group when divisible; the targeted resolver will map parsed 256 to 256/64/None only for marked primary INT8 converters. |
| 5 | Local regular-Hadamard weight rotation | `convert_to_quant/utils/convrot.py:build_hadamard`, `rotate_weight` | Original-domain `[out,in]` float32 weight and resolved group | For 64 and 256, normalized `H4` Kronecker matrices produce rotated-domain `W_rot = W @ H_block.T`; successful return is the only authority for applied-group state. |
| 6 | Phase 2 calibration preparation | `convert_to_quant/utils/tensor_utils.py:prepare_calibration_data` | Rotated weight, original-domain calibration `X`, actual group | Returns `X_rot = X @ H_block`, `Y_ref = X_rot @ W_rot.T`, and `H`; all are owned by the existing converter phase. |
| 7 | Initial row-wise quantization | `LearnedRoundingConverter._convert_int8_tensorwise`, `comfy.quant_ops.TensorWiseINT8Layout.quantize` | Rotated-domain `W_float32` | Clean `HEAD` returns per-row qdata and FP32 `[out,1]` scale. Only targeted primary simple ConvRot will use the specified local `absmax/127`, `1e-30`, divide, round, and `[-128,127]` helper. |
| 8 | Learned/AdaRound optimization or simple bypass | `LearnedRoundingConverter._convert_int8_tensorwise`, `_optimize_int8_adaround` | Initial qdata/scale plus rotated-domain calibration | Learned paths retain existing optimizer mathematics, including dualround; `no_learned_rounding` bypasses optimization without changing pipeline ownership. |
| 9 | Common dequantization and Phase 4 bias correction | `LearnedRoundingConverter._convert_int8_tensorwise`, `TensorWiseINT8Layout.dequantize` | Final qdata/scale, `X_rot`, and `Y_ref` | Common `qdata * scale` produces `dequantized_weight`; existing Phase 4 stores `bias_correction = mean(Y_ref - X_rot @ dequantized_weight.T)` in `extra_tensors`. |
| 10 | Tensor/metadata serialization and outer bias application | `convert_to_quant/formats/fp8_conversion.py:convert_to_fp8_scaled`, `utils/comfy_quant.py:create_comfy_quant_tensor` | Converter return tuple plus post-conversion applied-group state | Existing owners write weight, FP32 weight scale, `.comfy_quant`, optional header metadata, and apply ConvRot correction by addition or generic correction by subtraction. Public return types and bias ownership remain unchanged. |

### Mathematical comparison

| Operation | Clean local equation | Pinned/current comfy-kitchen equation | Required local change |
| --- | --- | --- | --- |
| Group-64 matrix construction | `H4` Kronecker power divided by `sqrt(64)` | Identical regular-Hadamard construction | None |
| Group-256 matrix construction | `H4` Kronecker power divided by `sqrt(256)` | Identical regular-Hadamard construction | None |
| Weight rotation | Grouped `W @ H.T` | Grouped `W @ H.T` | None |
| Activation rotation | Grouped `X @ H` | Grouped `X @ H` | None |
| Row absmax and stored scale | Clean layout uses `(row_absmax.clamp_min(1e-12))/127` | `(row_absmax.float()/127).clamp(min=1e-30)`, FP32 `[rows,1]` | Add the locked helper only for successfully rotated primary simple row-wise INT8. |
| Scale used for division | Clean layout uses reciprocal scale in tensor dtype; zero rows resolve through `1e-12` | Convert scale to tensor dtype/device and replace exact zero with `finfo(dtype).tiny` | Add the locked helper only in the same targeted branch. |
| Rounding and clamp | Clean layout clamps to `[-127,127]` before nearest rounding | Nearest round then clamp to `[-128,127]` | Add the locked helper only in the same targeted branch. |

The local Sylvester fallback for non-power-of-four groups differs from comfy-kitchen's rejection of those groups, but it is outside this task and remains protected. Groups 64 and 256 are mathematically identical across sources.

### Runtime and artifact contract

| Evidence | Verified value |
| --- | --- |
| 2688-wide model tensor key and shape | `blocks.0.adaln_proj.linear.weight`: `(96768, 2688)`, `torch.int8` |
| Its decoded `.comfy_quant` dictionary | `{'format':'int8_tensorwise','convrot':true,'convrot_groupsize':64}` |
| Deterministically selected group-256 comparison key and shape | Lexicographically first distinct compatible layer: `blocks.0.attn.out_proj.weight`, `(5376, 7168)`, `torch.int8` |
| Its decoded `.comfy_quant` dictionary | `{'format':'int8_tensorwise','convrot':true,'convrot_groupsize':256}` |
| Both `.weight_scale` shapes/dtypes | `(96768,1)` and `(5376,1)`, both `torch.float32` |
| File-header metadata keys | `_quantization_metadata` is absent; 1035 tensors and 250 per-layer `.comfy_quant` tensors were observed through `UnifiedSafetensorsLoader(..., low_memory=True)`. |
| Current ComfyUI load symbols/fields | `comfy/ops.py:_load_quantized_module` consumes `<prefix>weight`, `<prefix>weight_scale`, and `<prefix>comfy_quant`; `int8_tensorwise` reads top-level `convrot` and `convrot_groupsize` with legacy `params` fallback, default group 256, and errors on missing scale/unknown format. |
| Current ComfyUI runtime group propagation | `comfy/ops.py:linear_input_act` passes `params.convrot` and `params.convrot_groupsize` into `quant_ops.ck.int8_linear`; `_quantized_weight_state_dict` re-emits both top-level fields. Wrong/omitted group metadata therefore causes wrong/no activation rotation against the stored rotated weight. |

### Scope-preservation proof

| Protected area | Evidence it remains unchanged |
| --- | --- |
| FP8 | Resolver and applied-state hooks are gated by `fmt == "int8"`; existing FP8 branches and metadata are outside the allowlist. |
| Tensor-wise/block-wise INT8 | Resolver/arithmetic require row-wise ConvRot; tensor/block branches retain clean-HEAD selection and arithmetic. |
| Custom/fallback converters | New compatibility flag defaults false and is forced false for non-primary construction before overrides; no override dictionary can enable it. |
| Layer-config-owned behavior | Dynamic layer-config construction is not marked as the primary converter; its current override ownership remains unchanged. |
| Learned/AdaRound optimizer mathematics | New arithmetic helper requires `no_learned_rounding`; learned optimization receives the existing initial qdata/scale and rotated-domain calibration. |
| Calibration and bias correction | Existing Phase 2, common dequantization, Phase 4, outer correction owner, and signs remain in place and in order. |
| Public return tuple | Applied group is internal converter state; `(qdata, scale, dequantized, extra_tensors)` is unchanged. |
| `full_precision_matrix_mult` | Existing layer/global precedence and conditional serialization remain untouched. |
| `per_row` and `orig_dtype` metadata | Existing construction remains untouched; only truthful ConvRot fields/group are overridden post-conversion. |
| Filter selection | Generated filter flags continue to control inclusion before shared converter selection; resolver never reads filter identity. |
| Low-memory loading | Existing `UnifiedSafetensorsLoader(input_file, low_memory=low_memory)` and tensor-at-a-time loop remain unchanged. |

### Test delta and contradictions

| Field | Verified value |
| --- | --- |
| Existing tests proving preserved behavior | `test_build_hadamard`, `test_rotate_weight`, `test_rotate_activation`, `test_prepare_calibration_data`, `test_orthogonal_invariance`, `test_adaround_convrot_pipeline`, `test_triton_fused_kernel_compatibility`, `test_dynamic_convrot_resolution`, `test_dynamic_convrot_pipeline`, `test_int8_tensorwise`, `test_int8_rowwise`; `tests/test_cli_consistency.py` contains no parser-destination-named test functions and only the generic `get_arg_dests`, `get_arg_accesses`, `check_constants_filters`, and `analyze` checks. |
| Missing tests to add | `test_resolve_int8_convrot_group_size_exact_cases`; `test_quantize_convrot_int8_rowwise_equations`; `test_primary_simple_convrot_applied_groups_and_incompatible_width`; `test_primary_learned_convrot_group64_uses_optimizer`; `test_primary_simple_convrot_zero_rows`; `test_custom_and_fallback_convrot_do_not_use_primary_compat`; `test_convert_to_fp8_scaled_serializes_actual_primary_convrot_groups`; `test_convert_to_fp8_scaled_preserves_full_precision_and_header_group`. These assert every resolver case, exact qdata/FP32 `[rows,1]` scale for random/zero/group-64/group-256 inputs, 512→256, 2688→64, 96→no ConvRot, common dequantization/Phase 4, learned optimizer use without simple helper, custom/fallback isolation, truthful per-layer/header metadata, and full-precision preservation. |
| Unresolved contradictions | None |
| Re-analysis gate | PASS |

## Implementation Checklist

- [x] Rewrite the plan with the explicit user-decisions → clean-local-code → ComfyUI-consumer-contract → mathematical-comparison hierarchy.
- [x] Initialize from user/repository rules and inspect clean `HEAD` plus working-tree provenance in the required order.
- [x] Confirm the primary uses `low` reasoning, the subagents use `medium`, and all three analysis agents are spawned before primary source inspection.
- [x] Run the three read-only analysis tracks and independently verify their implementation-relevant source hunks in the primary agent.
- [x] Revalidate the model artifact through the mandated low-memory reads.
- [x] Inspect the pinned/current mathematical comparison sources and current ComfyUI consumer contract.
- [x] Fill every Re-analysis Evidence Ledger field and obtain `PASS` with `Unresolved contradictions` exactly `None`.
- [x] Revert the invalid dependency and replacement-branch draft without touching user files (the tracked tree was already clean, so no restore was required).
- [x] Implement the primary static 256-to-64 resolver.
- [x] Implement the specified local arithmetic at the existing simple-mode quantization point only.
- [x] Record actual local ConvRot success and serialize it for primary simple and learned/AdaRound paths.
- [x] Update only INT8 ConvRot documentation.
- [x] Add focused resolver, arithmetic, pipeline, scope-regression, and metadata tests.
- [x] Create task-only checkpoint commits after each verified changed state.
- [x] Run focused validation and audit every final diff hunk against this plan.
- [x] Autosquash the contiguous checkpoint series and verify identical tree content, one concise feature commit, and no unrelated changes.
