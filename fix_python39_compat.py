#!/usr/bin/env python3
"""
Script to fix Python 3.9 compatibility issues by converting union types from | to Union[...]
"""

import re
import sys
from pathlib import Path


def has_union_operator(content):
    """Check if the file has union type operators in type annotations"""
    # Pattern to match type annotations with | operator
    # This pattern tries to avoid matching bitwise OR operations
    patterns = [
        r':\s*\w+\s*\|\s*\w+',  # variable: type1 | type2
        r'->\s*\w+\s*\|\s*\w+',  # -> type1 | type2
        r'\[\s*\w+\s*\|\s*\w+',  # [type1 | type2
    ]
    for pattern in patterns:
        if re.search(pattern, content):
            return True
    return False


def has_union_import(content):
    """Check if Union is already imported from typing"""
    return re.search(r'from typing import.*\bUnion\b', content)


def add_union_import(content):
    """Add Union to typing imports if not present"""

    # If Union is already imported, return unchanged
    if has_union_import(content):
        return content

    # Find existing typing import
    typing_import_match = re.search(r'^from typing import (.+)$', content, re.MULTILINE)

    if typing_import_match:
        # Add Union to existing import
        imports = typing_import_match.group(1)
        # Check if it's a multi-line import
        if '(' in imports:
            # Multi-line import - add Union before the closing paren
            content = re.sub(
                r'(from typing import[^)]+)\)',
                r'\1, Union)',
                content,
                count=1
            )
        else:
            # Single line import - add Union to the list
            new_imports = imports.rstrip() + ', Union'
            content = re.sub(
                r'^from typing import .+$',
                f'from typing import {new_imports}',
                content,
                count=1,
                flags=re.MULTILINE
            )
    else:
        # No typing import exists - add one after other imports
        # Find the last import statement
        import_lines = list(re.finditer(r'^(import |from \w+ import)', content, re.MULTILINE))
        if import_lines:
            last_import = import_lines[-1]
            insert_pos = content.find('\n', last_import.end()) + 1
            content = content[:insert_pos] + '\nfrom typing import Union\n' + content[insert_pos:]
        else:
            # No imports at all - add at the beginning
            content = 'from typing import Union\n\n' + content

    return content


def convert_union_operators(content):
    """Convert | union operators to Union[...] syntax"""

    # This is complex because we need to handle nested types and multiple unions
    # We'll use a more targeted approach

    # Pattern 1: Simple unions in type hints (param: Type1 | Type2)
    # Match type annotations with union operators
    def replace_union(match):
        before = match.group(1)  # Everything before the types
        types_str = match.group(2)  # The types with | operators

        # Split by | and clean up whitespace
        types = [t.strip() for t in types_str.split('|')]

        # Create Union syntax
        union_str = f'Union[{", ".join(types)}]'

        return f'{before}{union_str}'

    # Pattern for parameter type hints: name: type1 | type2 | ... =
    content = re.sub(
        r'(\w+:\s*)([\w\[\],\s]+(?:\s*\|\s*[\w\[\],\s]+)+)(\s*[=,\)])',
        lambda m: f'{m.group(1)}Union[{", ".join([t.strip() for t in m.group(2).split("|")])}]{m.group(3)}',
        content
    )

    # Pattern for return type hints: -> type1 | type2
    content = re.sub(
        r'(->\s*)([\w\[\],\s]+(?:\s*\|\s*[\w\[\],\s]+)+)(\s*:)',
        lambda m: f'{m.group(1)}Union[{", ".join([t.strip() for t in m.group(2).split("|")])}]{m.group(3)}',
        content
    )

    # Pattern for variable annotations: var: type1 | type2 =
    content = re.sub(
        r'(:\s*)([\w\[\],\s]+(?:\s*\|\s*[\w\[\],\s]+)+)(\s*=)',
        lambda m: f'{m.group(1)}Union[{", ".join([t.strip() for t in m.group(2).split("|")])}]{m.group(3)}',
        content
    )

    return content


def fix_file(filepath):
    """Fix union type operators in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Check if file needs fixing
        if not has_union_operator(content):
            return False

        # Add Union import if needed
        content = add_union_import(content)

        # Convert union operators
        content = convert_union_operators(content)

        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return False


def main():
    """Main function to fix all Python files"""
    # Find all Python files in rl4co directory
    base_dir = Path(__file__).parent

    python_files = []
    for pattern in ['rl4co/**/*.py', 'ppo_loop/*.py']:
        python_files.extend(base_dir.glob(pattern))

    fixed_count = 0
    for filepath in python_files:
        if fix_file(filepath):
            print(f"Fixed: {filepath}")
            fixed_count += 1

    print(f"\nTotal files fixed: {fixed_count}")


if __name__ == '__main__':
    main()
