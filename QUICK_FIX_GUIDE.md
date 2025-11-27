# Quick Fix Guide for Linux Setup

## Problem
Getting `ModuleNotFoundError: No module named 'rl4co'` error when trying to run `meta_train.py`.

## Root Cause
The `inner_loop` package has some files that import `rl4co` directly instead of `inner_loop.rl4co`. This happened because the codebase was originally just `rl4co`, and when it was moved into `inner_loop/`, not all imports were updated.

## Solution (Run on Linux Machine)

### Step 1: Run the automated fix script
```bash
cd /path/to/rl4co\ git
python fix_imports.py
```

This will automatically fix all incorrect imports in the `inner_loop` package.

**Expected output:**
```
================================================================================
Import Fixer for inner_loop Package
================================================================================
Searching in: /path/to/rl4co git/inner_loop
Found XXX Python files to check

  ✓ Fixed: rl4co/utils/trainer.py
  ✓ Fixed: rl4co/utils/meta_trainer.py
  ✓ Fixed: rl4co/tasks/train.py

================================================================================
✅ Fixed imports in 3 file(s):
   - rl4co/utils/trainer.py
   - rl4co/utils/meta_trainer.py
   - rl4co/tasks/train.py
================================================================================
```

### Step 2: Verify the fix
```bash
cd ppo_loop
python -c "from dvrp_env import DVRPEnv; print('✅ Success!')"
```

### Step 3: Run training with correct device argument
```bash
# For CPU training
python meta_train.py --episodes 12800 --device cpu

# For GPU training (use 'cuda' not 'gpu')
python meta_train.py --episodes 12800 --device cuda

# Full example with all options
python meta_train.py \
    --episodes 12800 \
    --num-envs 64 \
    --policy-update-interval 3 \
    --device cuda \
    --save-interval 100
```

## Important Notes

### ⚠️ Device Argument
Use `--device cuda` for GPU, NOT `--device gpu`!

**Correct:**
```bash
python meta_train.py --episodes 1000 --device cuda  ✅
```

**Wrong:**
```bash
python meta_train.py --episodes 1000 --device gpu   ❌
# This will give: error: argument --device: invalid choice: 'gpu'
```

### Valid device options:
- `cpu` - Run on CPU
- `cuda` - Run on GPU (if CUDA is available)

## Verification Checklist

Before running training, verify:

1. ✅ `inner_loop` directory exists
   ```bash
   ls -la "rl4co git/inner_loop"
   ```

2. ✅ Import fix script ran successfully
   ```bash
   python fix_imports.py
   ```

3. ✅ Can import DVRPEnv
   ```bash
   cd ppo_loop
   python -c "from dvrp_env import DVRPEnv; print('OK')"
   ```

4. ✅ CUDA is available (for GPU training)
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

## What the Fix Script Does

The script searches for and fixes these problematic import patterns:

1. `from rl4co import utils` → `from inner_loop.rl4co import utils`
2. `from rl4co.envs import ...` → `from inner_loop.rl4co.envs import ...`
3. `import rl4co` → `import inner_loop.rl4co`

Files that are typically fixed:
- `inner_loop/rl4co/utils/trainer.py`
- `inner_loop/rl4co/utils/meta_trainer.py`
- `inner_loop/rl4co/tasks/train.py`

## Still Getting Errors?

If you still see import errors after running `fix_imports.py`, try:

1. **Clear Python cache:**
   ```bash
   find "rl4co git" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
   find "rl4co git" -type f -name "*.pyc" -delete 2>/dev/null
   ```

2. **Run the fix script again:**
   ```bash
   python fix_imports.py
   ```

3. **Check for other import patterns:**
   ```bash
   cd inner_loop
   grep -r "^from rl4co" --include="*.py" | grep -v "inner_loop"
   ```

   If this returns any results, there are more imports that need fixing.

## Complete Setup Commands for Linux

```bash
# 1. Navigate to project directory
cd /home/jiangwolin/"rl4co git"

# 2. Fix imports
python fix_imports.py

# 3. Check setup
cd ppo_loop
python setup_package.py

# 4. Verify imports work
python -c "from dvrp_env import DVRPEnv; print('✅ Setup complete!')"

# 5. Start training
python meta_train.py --episodes 12800 --device cuda
```

## Need More Help?

See:
- `TROUBLESHOOTING.md` for detailed debugging
- `README_SETUP.md` for complete setup instructions
