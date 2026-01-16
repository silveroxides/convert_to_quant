
import ast
import os
import sys

def get_arg_dests(parser_path):
    with open(parser_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    
    dests = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # rudimentary check for add_argument calls
             if hasattr(node.func, 'attr') and node.func.attr == 'add_argument':
                # iterate keywords to find dest
                for kw in node.keywords:
                    if kw.arg == 'dest':
                        if isinstance(kw.value, ast.Constant):
                            dests.add(kw.value.value)
    
    # Also manual list of args typically found in ADD_ARGUMENT calls manually
    # ideally we import the module, but that's risky if it has side effects.
    # Let's try basic AST regex-ish approach first or just "dest="
    
    return dests

def get_arg_accesses(main_path):
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # regex for args.NAME
    import re
    accesses = set(re.findall(r'args\.([a-zA-Z0-9_]+)', content))
    return accesses

def check_constants_filters():
    # Load constants to get MODEL_FILTERS keys
    # Load constants to get MODEL_FILTERS keys
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
    from convert_to_quant.constants import MODEL_FILTERS
    return set(MODEL_FILTERS.keys())

def analyze():
    # Base dir is ../convert_to_quant relative to this script
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "convert_to_quant")
    parser_path = os.path.join(base_dir, "cli", "argument_parser.py")
    main_path = os.path.join(base_dir, "cli", "main.py")
    
    # 1. Parse argument_parser.py to find definitions
    # Since it's complex, we might just grep for dest= or strings?
    # Actually, let's use a simpler heuristic: read the file content and look for 'dest='
    
    with open(parser_path, 'r') as f:
        parser_content = f.read()
    
    import re
    defined_args = set(re.findall(r'dest="([^"]+)"', parser_content)) 
    defined_args.update(re.findall(r"dest='([^']+)'", parser_content))
    
    # Add manual ones that might be missed (like implicit dest from flags)
    # This is a limitation of static analysis without full import
    
    # 2. Add filter args from constants
    try:
        filters = check_constants_filters()
        defined_args.update(filters)
        print(f"Found {len(filters)} dynamic filter arguments.")
    except Exception as e:
        print(f"Warning: Could not load constants: {e}")

    # 3. Find usages in main.py
    accessed_args = get_arg_accesses(main_path)
    
    print(f"Defined Arguments (Approx {len(defined_args)}): {sorted(list(defined_args))}")
    print(f"Accessed Arguments (Approx {len(accessed_args)}): {sorted(list(accessed_args))}")
    
    # 4. Check for unused
    unused = defined_args - accessed_args
    # Filter out known false positives
    unused = {u for u in unused if u not in ["help", "version"]}
    
    print("\n" + "="*50)
    print("POTENTIALLY UNUSED ARGUMENTS (Defined but not accessed in main.py):")
    print("="*50)
    if unused:
        for u in sorted(list(unused)):
            print(f" - {u}")
    else:
        print("None found!")

if __name__ == "__main__":
    analyze()
