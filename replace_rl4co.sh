#!/bin/bash

# Directory to search (default: current directory)
TARGET_DIR="${1:-.}"

echo "Replacing 'from rl4co.' with 'from inner_loop.rl4co.' in directory: $TARGET_DIR"

# Find all .py files and apply an in-place sed replacement
find "$TARGET_DIR" -type f -name "*.py" | while read -r file; do
    echo "Processing: $file"
    sed -i '' 's/from rl4co\./from inner_loop.rl4co./g' "$file"
done

echo "Done!"
