#!/usr/bin/env python3
"""
Automated script to fix incorrect imports in inner_loop package.

This fixes imports that reference 'rl4co' directly instead of 'inner_loop.rl4co'.
"""

from pathlib import Path
import re
import sys

def fix_imports_in_file(file_path: Path) -> bool:
    """Fix imports in a single file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content

        # Fix: from rl4co import ... -> from inner_loop.rl4co import ...
        content = re.sub(
            r'^from rl4co import',
            'from inner_loop.rl4co import',
            content,
            flags=re.MULTILINE
        )

        # Fix: from rl4co. -> from inner_loop.rl4co.
        content = re.sub(
            r'^from rl4co\.',
            'from inner_loop.rl4co.',
            content,
            flags=re.MULTILINE
        )

        # Fix: import rl4co -> import inner_loop.rl4co
        content = re.sub(
            r'^import rl4co$',
            'import inner_loop.rl4co',
            content,
            flags=re.MULTILINE
        )

        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True
        return False

    except Exception as e:
        print(f"  ⚠️  Error processing {file_path}: {e}")
        return False


def main():
    """Fix all import issues in inner_loop package."""

    # Get project root
    script_dir = Path(__file__).resolve().parent
    inner_loop_path = script_dir / "inner_loop"

    print("=" * 80)
    print("Import Fixer for inner_loop Package")
    print("=" * 80)
    print(f"\nSearching in: {inner_loop_path}")

    if not inner_loop_path.exists():
        print(f"\n❌ ERROR: inner_loop directory not found at {inner_loop_path}")
        print("Make sure you run this script from the project root directory.")
        return 1

    # Find all Python files in inner_loop
    python_files = list(inner_loop_path.rglob("*.py"))

    # Exclude checkpoint and cache directories
    python_files = [
        f for f in python_files
        if '.ipynb_checkpoints' not in str(f)
        and '__pycache__' not in str(f)
    ]

    print(f"Found {len(python_files)} Python files to check\n")

    # Fix imports
    fixed_files = []
    for file_path in python_files:
        rel_path = file_path.relative_to(inner_loop_path)
        if fix_imports_in_file(file_path):
            print(f"  ✓ Fixed: {rel_path}")
            fixed_files.append(rel_path)

    # Summary
    print("\n" + "=" * 80)
    if fixed_files:
        print(f"✅ Fixed imports in {len(fixed_files)} file(s):")
        for f in fixed_files:
            print(f"   - {f}")
    else:
        print("✅ No import issues found - all files are correct!")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
