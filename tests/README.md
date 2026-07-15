# Test suites

The default pytest boundary is this directory. The checkout under
`temp-reference/` is reference material and is intentionally not collected by
normal project test runs.

```powershell
# Fast contract checks
python -m pytest -m "unit and not slow" -q

# Complete project suite
python -m pytest -q

# Explicit reference suite (requires comfy-kitchen build dependencies)
python -m pytest temp-reference/comfy-kitchen-main/tests -q
```

Markers distinguish fast unit contracts, safetensors/conversion integration
tests, CUDA-only coverage, and slow optimization-loop characterization.
