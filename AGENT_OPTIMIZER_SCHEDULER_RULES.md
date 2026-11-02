# 🛑 AI AGENT RULES: CUSTOM OPTIMIZERS & SCHEDULERS

**CRITICAL WARNING FOR ALL FUTURE AI AGENTS:**
Do **NOT** implement standard PyTorch training loops (`model.train()`, `scheduler.step()`, etc.) in this codebase.

This project uses highly specialized, custom-built manual optimizer and scheduler loops designed for ultra-low-precision quantization via SVD projection and Continuous Relaxation (AdaRound). These loops contain complex heuristics, shape-aware adjustments, and specific counter-management logic that **must be replicated 1:1** when adding new formats or optimization paths.

If you fail to follow these rules, you will break the learning rate decay curves, ruin early stopping, and cause massive accuracy regressions.

---

## 1. The "1:1 Parity" Golden Rule
Any new `_optimize_*` method (e.g., for a new format, new optimizer, or new quantization approach like ConvRot/AdaRound) **MUST** perfectly copy the state management, progress bar updating, early stopping checks, and learning rate scheduling mechanics of the existing `_optimize_original` method.

You cannot write a "simplified" loop. You must implement the full loop mechanics step-by-step as outlined below.

---

## 2. Pre-Loop Initialization

Before the `for i in pbar:` loop even starts, you MUST initialize shape-aware parameters for the plateau scheduler using the tensor dimensions `(M, N)`:

```python
effective_patience, effective_factor, effective_cooldown = self._compute_shape_aware_plateau_params(M, N)
```

---

## 3. Step-by-Step Loop Mechanics

Every optimization loop inside a converter (like `LearnedRoundingConverter`) must follow this exact structure inside the `for i in pbar:` loop:

### Step 3.1: The Gradient Update
Whether using a PyTorch Optimizer (AdamW, Prodigy) or analytical SVD gradients (`original`), the forward/backward pass must happen *first*.

### Step 3.2: Capture Previous State & Improvement Check
You must capture the `worse_loss_counter` **before** any resets happen. Then, you must NOT simply use `current_loss < best_loss`. You **must** use the inherited threshold checking method:
```python
current_loss_val = loss.item()

# MUST BE CAPTURED BEFORE IMPROVEMENT CHECK!
prev_worse_counter = worse_loss_counter

improved = self._check_improvement(current_loss_val, best_loss)
```

### Step 3.3: Counter Management (THE MOST COMMON AI MISTAKE)
AI agents almost always mess this up. There are two counters: `plateau_counter` and `worse_loss_counter`.
*   `plateau_counter` is ALWAYS reset on improvement.
*   `worse_loss_counter` is **ONLY** reset on improvement if `lr_adaptive_mode == "simple-reset"`. If it is `"no-reset"`, the counter is preserved so the adaptive cosine scheduler knows exactly how long the current "stall" tier has been ongoing.

**Correct Implementation:**
```python
if improved:
    best_loss = current_loss_val
    best_tensor = tensor.detach().clone() # Or best_delta, best_V
    plateau_counter = 0

    # CRITICAL: Do NOT unconditionally reset worse_loss_counter here!
    if self.lr_adaptive_mode == "simple-reset":
        worse_loss_counter = 0
    # no-reset mode: worse_loss_counter preserved for tier calculation
else:
    worse_loss_counter += 1
    plateau_counter += 1
```

### Step 3.4: Manual LR Scheduling
You **must** implement all three schedules (`exponential`, `plateau`, `adaptive`) manually using `if/elif/else`. You must include all `debug` logging exactly as shown.

*(Note: If using standard PyTorch optimizers like AdamW/Prodigy, you update `param_group["lr"]`. If implementing a gradient-only optimizer like `original`, you just track `curr_lr` variable).*

**Correct Implementation (PyTorch Optimizer style):**
```python
if schedule_name == "exponential":
    curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
    if optimizer is not None:
        for param_group in optimizer.param_groups: param_group["lr"] = curr_lr

elif schedule_name == "plateau":
    # Plateau requires shape-aware effective_patience/factor/cooldown
    if cooldown_counter > 0:
        cooldown_counter -= 1
        debug(f"      [LR] Cooldown: {cooldown_counter} left")
    elif plateau_counter >= effective_patience:
        debug(f"      [LR] Plateau {plateau_counter}/{effective_patience} reached. Decaying.")
        if curr_lr > self.lr_min:
            old_lr = curr_lr
            curr_lr = max(curr_lr * effective_factor, self.lr_min)
            if optimizer is not None:
                for param_group in optimizer.param_groups: param_group["lr"] = curr_lr
            cooldown_counter = effective_cooldown
            debug(f"      [LR] Decay: {old_lr:.2e} -> {curr_lr:.2e} (Factor: {effective_factor:.4f})")
        plateau_counter = 0
    else:
        if plateau_counter > 0:
            debug(f"      [LR] Waiting: {plateau_counter}/{effective_patience} (Loss: {current_loss_val:.3e})")

else:  # "adaptive" cosine-based schedule
    # MUST use counter_for_update to prevent compounding
    counter_for_update = prev_worse_counter if improved else worse_loss_counter
    new_lr, lr_updated = self._adaptive_lr_update_cosine(curr_lr, improved, counter_for_update, i, (M, N), self.early_stop_lr)
    if lr_updated:
        curr_lr = new_lr
        if optimizer is not None:
            for param_group in optimizer.param_groups: param_group["lr"] = curr_lr

    # Reset happens HERE for no-reset mode
    if improved and self.lr_adaptive_mode == "no-reset":
        worse_loss_counter = 0
```

### Step 3.5: Dynamic Progress Bar Postfixes
The user relies heavily on the CLI output to monitor optimization. The postfix must reflect the active schedule.

**Correct Implementation:**
```python
if schedule_name == "plateau":
    pbar.set_postfix({"loss": f"{current_loss_val:.3e}", "best": f"{best_loss:.3e}", "lr": f"{curr_lr:.2e}", "plateau": f"{plateau_counter}/{effective_patience}"})
else:
    pbar.set_postfix({"loss": f"{current_loss_val:.3e}", "best": f"{best_loss:.3e}", "lr": f"{curr_lr:.2e}", "worse_count": f"{worse_loss_counter}"})
```

### Step 3.6: Informative Early Stopping
Do not just `break`. You must evaluate the comprehensive early stopping conditions and print the precise `info()` message indicating *why* the loop stopped. Use the gold standard combo-conditions found in `_optimize_original`.

**Correct Implementation:**
```python
# Note: Use `current_loss_val` or `best_loss` depending on whether the optimizer allows jitter.
# For exact parity with _optimize_original, current_loss_val is typically checked.
if current_loss_val <= self.early_stop_loss or curr_lr <= self.early_stop_lr or worse_loss_counter > self.early_stop_stall:
    if curr_lr <= self.early_stop_lr * 1.75 and worse_loss_counter > self.early_stop_stall * 0.95:
        info("\n      - Loss has stalled and learning rate has bottomed out. Stopping.")
    elif current_loss_val <= self.early_stop_loss and curr_lr <= self.early_stop_lr * 1.75:
        info("\n      - Learning Rate has bottomed out and loss is negligible. Stopping.")
    elif worse_loss_counter > self.early_stop_stall * 0.95 and current_loss_val > self.early_stop_loss * 2:
        info("\n      - Loss is negligible and loss has stalled. Stopping.")
    elif current_loss_val <= self.early_stop_loss:
        info("\n      - Loss is negligible. Stopping.")
    elif curr_lr <= self.early_stop_lr:
        info("\n      - Learning Rate has bottomed out. Stopping.")
    elif worse_loss_counter > self.early_stop_stall:
        info("\n      - Loss has stalled. Stopping.")
    break
```

---

## 4. Checklist for New Optimizer Code
Before finalizing any code mode session involving `convert_to_quant` optimization loops, verify:
- [ ] Did you initialize `effective_patience`, `effective_factor`, and `effective_cooldown` before the loop?
- [ ] Did you capture `prev_worse_counter = worse_loss_counter` BEFORE checking `improved`?
- [ ] Did you use `self._check_improvement` instead of `<`?
- [ ] Did you wrap the `worse_loss_counter = 0` reset inside `if self.lr_adaptive_mode == "simple-reset":` during the `improved` block?
- [ ] Did you implement all 3 schedulers (`exponential`, `plateau`, `adaptive`) exactly as shown?
- [ ] Did you include the `debug` logs for `cooldown`, `Plateau reached`, `Decay`, and `Waiting` in the plateau scheduler?
- [ ] Did you include the schedule-aware dynamic postfix?
- [ ] Did you implement the comprehensive 6-condition early stopping `info()` logs?
- [ ] If using Prodigy, did you instantiate it without `split_groups=True` to avoid extraneous CLI warnings?

If the answer to any of these is NO, **rewrite the loop** before committing the changes.