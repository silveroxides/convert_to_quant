
import os
import subprocess
import sys

def run_test(name, cmd_args, expected_file=None):
    print(f"\n[TEST] {name}")
    # Use the installed CLI command provided by pip install -e .
    base_cmd = ["convert_to_quant"]
    full_cmd = base_cmd + cmd_args
    print(f"Command: {' '.join(full_cmd)}")
    
    try:
        # shell=True might be needed on Windows to find the shim in Scripts
        subprocess.run(full_cmd, check=True, shell=True)
        print(">> SUCCESS (Run)")

        if expected_file:
            if os.path.exists(expected_file):
                print(f">> SUCCESS (File created: {expected_file})")
            else:
                print(f">> FAILURE (File missing: {expected_file})")
                return False
    except subprocess.CalledProcessError as e:
        print(f">> FAILURE (Error code {e.returncode})")
        return False
    return True

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run functional regression tests.")
    parser.add_argument("input_file", nargs="?", help="Path to input safetensors file")
    args = parser.parse_args()

    if args.input_file:
        input_file = args.input_file
    else:
        # Default: Input file is in the root directory (parent of tests/)
        input_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "flux2-dev-testlayers.safetensors")

    if not os.path.exists(input_file):
        print(f"Error: Test file '{input_file}' not found.")
        print("Usage: python tests/test_functional.py [input_file]")
        return
    
    print(f"Using input file: {input_file}")

    tests = [
        # 1. FP8 Simple
        {
            "name": "FP8 Simple",
            "args": ["-i", input_file, "--simple", "--output", "out_fp8_simple.safetensors"],
            "expected": "out_fp8_simple.safetensors"
        },
        # 2. FP8 Learned (1 iter)
        {
            "name": "FP8 Learned",
            "args": ["-i", input_file, "--num_iter", "1", "--output", "out_fp8_learned.safetensors"],
            "expected": "out_fp8_learned.safetensors"
        },
        # 3. INT8
        {
            "name": "INT8 Simple",
            "args": ["-i", input_file, "--int8", "--simple", "--block_size", "128", "--output", "out_int8.safetensors"],
            "expected": "out_int8.safetensors"
        },
        # 4. NVFP4 Simple
        {
            "name": "NVFP4 Simple",
            "args": ["-i", input_file, "--nvfp4", "--simple", "--output", "out_nvfp4_simple.safetensors"],
            "expected": "out_nvfp4_simple.safetensors"
        },
        # 5. NVFP4 Learned
        {
            "name": "NVFP4 Learned",
            "args": ["-i", input_file, "--nvfp4", "--num_iter", "1", "--output", "out_nvfp4_learned.safetensors"],
            "expected": "out_nvfp4_learned.safetensors"
        },
        # 6. Filter Check (Flux2) - should run without error
        {
            "name": "Filter Flux2",
            "args": ["-i", input_file, "--flux2", "--simple", "--output", "out_flux2.safetensors"],
            "expected": "out_flux2.safetensors"
        },
        # 7. Custom Layers
        {
            "name": "Custom Layers",
            "args": ["-i", input_file, "--simple", "--custom-layers", "transformer", "--custom-type", "int8", "--custom-block-size", "128", "--output", "out_custom.safetensors"],
            "expected": "out_custom.safetensors"
        },
        # 8. Logging - Minimal
        {
            "name": "Log Minimal",
            "args": ["-i", input_file, "--simple", "--verbose", "MINIMAL", "--output", "out_log_min.safetensors"],
            "expected": "out_log_min.safetensors"
        },
        # 9. Logging - Verbose
        {
            "name": "Log Verbose",
            "args": ["-i", input_file, "--simple", "--verbose", "VERBOSE", "--output", "out_log_verb.safetensors"],
            "expected": "out_log_verb.safetensors"
        },
        # 10. Logging - Debug
        {
            "name": "Log Debug",
            "args": ["-i", input_file, "--simple", "--verbose", "DEBUG", "--output", "out_log_debug.safetensors"],
            "expected": "out_log_debug.safetensors"
        }
    ]

    passed = 0
    for t in tests:
        if run_test(t["name"], t["args"], t["expected"]):
            passed += 1
    
    print(f"\nSummary: {passed}/{len(tests)} tests passed.")
    
    # Cleanup
    print("\nCleaning up output files...")
    for t in tests:
        if t["expected"] and os.path.exists(t["expected"]):
            os.remove(t["expected"])

if __name__ == "__main__":
    main()
