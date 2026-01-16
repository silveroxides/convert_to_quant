#!/usr/bin/env python3
"""
AST Analysis Script for convert_to_quant.py

Analyzes the code structure to map:
- Function definitions with line counts
- Class definitions with methods
- Import dependencies
- Suggested module groupings
"""
import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class CodeAnalyzer(ast.NodeVisitor):
    """Analyze Python source code structure using AST."""
    
    def __init__(self):
        self.functions: List[Dict] = []
        self.classes: List[Dict] = []
        self.imports: List[str] = []
        self.from_imports: List[Tuple[str, List[str]]] = []
        self.current_class: Optional[str] = None
    
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        names = [alias.name for alias in node.names]
        self.from_imports.append((module, names))
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self.current_class is None:
            # Top-level function
            self.functions.append({
                "name": node.name,
                "start_line": node.lineno,
                "end_line": node.end_lineno or node.lineno,
                "lines": (node.end_lineno or node.lineno) - node.lineno + 1,
                "docstring": ast.get_docstring(node),
                "args": [arg.arg for arg in node.args.args],
            })
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        methods = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append({
                    "name": child.name,
                    "start_line": child.lineno,
                    "end_line": child.end_lineno or child.lineno,
                    "lines": (child.end_lineno or child.lineno) - child.lineno + 1,
                })
        
        self.classes.append({
            "name": node.name,
            "start_line": node.lineno,
            "end_line": node.end_lineno or node.lineno,
            "lines": (node.end_lineno or node.lineno) - node.lineno + 1,
            "methods": methods,
            "docstring": ast.get_docstring(node),
        })
        
        # Don't recurse into class - we already extracted methods
        # self.generic_visit(node)


def analyze_file(filepath: Path) -> Dict:
    """Analyze a Python file and return its structure."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source)
    
    analyzer = CodeAnalyzer()
    analyzer.visit(tree)
    
    total_lines = source.count("\n") + 1
    
    return {
        "filepath": str(filepath),
        "total_lines": total_lines,
        "imports": analyzer.imports,
        "from_imports": analyzer.from_imports,
        "functions": analyzer.functions,
        "classes": analyzer.classes,
    }


def print_report(analysis: Dict) -> None:
    """Print a human-readable analysis report."""
    print(f"\n{'='*80}")
    print(f"FILE ANALYSIS: {analysis['filepath']}")
    print(f"{'='*80}")
    print(f"Total lines: {analysis['total_lines']}")
    
    # Imports
    print(f"\n--- IMPORTS ({len(analysis['imports']) + len(analysis['from_imports'])}) ---")
    for imp in analysis["imports"]:
        print(f"  import {imp}")
    for module, names in analysis["from_imports"]:
        print(f"  from {module} import {', '.join(names)}")
    
    # Classes
    print(f"\n--- CLASSES ({len(analysis['classes'])}) ---")
    for cls in sorted(analysis["classes"], key=lambda x: -x["lines"]):
        print(f"\n  {cls['name']} (lines {cls['start_line']}-{cls['end_line']}, {cls['lines']} total)")
        if cls["docstring"]:
            print(f"    Doc: {cls['docstring'][:80]}...")
        print(f"    Methods ({len(cls['methods'])}):")
        for method in sorted(cls["methods"], key=lambda x: -x["lines"]):
            print(f"      - {method['name']}: {method['lines']} lines ({method['start_line']}-{method['end_line']})")
    
    # Functions
    print(f"\n--- TOP-LEVEL FUNCTIONS ({len(analysis['functions'])}) ---")
    for fn in sorted(analysis["functions"], key=lambda x: -x["lines"]):
        print(f"  {fn['name']}: {fn['lines']} lines ({fn['start_line']}-{fn['end_line']})")
        if fn["args"]:
            print(f"    Args: {', '.join(fn['args'][:5])}{'...' if len(fn['args']) > 5 else ''}")
    
    # Summary for refactoring
    print(f"\n--- REFACTORING SUMMARY ---")
    large_funcs = [f for f in analysis["functions"] if f["lines"] > 100]
    large_classes = [c for c in analysis["classes"] if c["lines"] > 200]
    
    print(f"  Large functions (>100 lines): {len(large_funcs)}")
    for fn in large_funcs:
        print(f"    - {fn['name']}: {fn['lines']} lines")
    
    print(f"  Large classes (>200 lines): {len(large_classes)}")
    for cls in large_classes:
        print(f"    - {cls['name']}: {cls['lines']} lines")
    
    # Suggested modules
    print(f"\n--- SUGGESTED MODULE GROUPINGS ---")
    print("  constants.py: All *_AVOID_KEY_NAMES, *_LAYER_KEYNAMES constants")
    print("  converters/learned_rounding.py: LearnedRoundingConverter class")
    print("  cli/argument_parser.py: MultiHelpArgumentParser class")
    print("  cli/main.py: main() function")
    print("  formats/fp8_conversion.py: convert_to_fp8_scaled()")
    print("  formats/format_migration.py: convert_fp8_scaled_to_comfy_quant()")
    print("  formats/int8_conversion.py: convert_int8_to_comfy_quant()")
    print("  formats/legacy_utils.py: add_legacy_input_scale(), cleanup_fp8_scaled()")
    print("  config/layer_config.py: load_layer_config(), get_layer_settings(), generate_config_template()")
    print("  utils/tensor_utils.py: dict_to_tensor(), tensor_to_dict(), normalize_tensorwise_scales()")
    print("  utils/comfy_quant.py: create_comfy_quant_tensor(), edit_comfy_quant(), parse_add_keys_string()")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        # Default to analyzing convert_to_quant.py
        target = Path(__file__).parent / "convert_to_quant_refactor" / "convert_to_quant.py"
        if not target.exists():
            target = Path(__file__).parent / "convert_to_quant" / "convert_to_quant.py"
    else:
        target = Path(sys.argv[1])
    
    if not target.exists():
        print(f"ERROR: File not found: {target}")
        sys.exit(1)
    
    analysis = analyze_file(target)
    print_report(analysis)


if __name__ == "__main__":
    main()
