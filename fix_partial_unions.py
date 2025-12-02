#!/usr/bin/env python3
"""Fix partially converted union types"""
import re
from pathlib import Path

# List of files with partial union conversions
files_to_fix = [
    "rl4co/models/zoo/eas/search.py",
    "rl4co/models/zoo/active_search/search.py",
    "rl4co/envs/graph/flp/generator.py",
    "rl4co/envs/graph/mcp/generator.py",
    "rl4co/envs/routing/pctsp/generator.py",
    "rl4co/envs/routing/tsp/generator.py",
    "rl4co/envs/routing/mpdp/generator.py",
    "rl4co/envs/routing/op/generator.py",
    "rl4co/envs/routing/svrp/generator.py",
    "rl4co/envs/routing/atsp/generator.py",
    "rl4co/envs/routing/cvrp/generator.py",
    "rl4co/envs/routing/cvrptw/generator.py",
    "rl4co/envs/routing/mtvrp/generator.py",
    "rl4co/envs/routing/pdp/generator.py",
    "rl4co/envs/routing/mtsp/generator.py",
]

def fix_partial_unions(content):
    """Fix partially converted union types like Union[A, B] | C | D"""

    # Pattern to match Union[...] | ... | ...
    pattern = r'Union\[([\w\[\],\s]+)\]\s*\|\s*([\w\[\],\s\|]+)'

    def replace_union(match):
        # Get the types inside Union
        inside_union = match.group(1)
        # Get the types after the |
        after_pipe = match.group(2)

        # Split the after_pipe by | to get individual types
        remaining_types = [t.strip() for t in after_pipe.split('|')]

        # Combine all types
        all_types = [t.strip() for t in inside_union.split(',')] + remaining_types

        return f'Union[{", ".join(all_types)}]'

    # Keep replacing until no more matches
    prev_content = None
    while prev_content != content:
        prev_content = content
        content = re.sub(pattern, replace_union, content)

    return content

def main():
    base_dir = Path(__file__).parent
    fixed_count = 0

    for file_path in files_to_fix:
        full_path = base_dir / file_path
        if not full_path.exists():
            print(f"Skipping {file_path} (not found)")
            continue

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original = content
            content = fix_partial_unions(content)

            if content != original:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Fixed: {file_path}")
                fixed_count += 1
            else:
                print(f"No changes needed: {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"\nTotal files fixed: {fixed_count}")

if __name__ == '__main__':
    main()
