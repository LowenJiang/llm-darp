# Troubleshooting: ModuleNotFoundError: No module named 'inner_loop'

## Quick Fix

Run this diagnostic script first:
```bash
cd /path/to/rl4co\ git/ppo_loop
python setup_package.py
```

This will:
1. Check if `inner_loop` directory exists
2. Verify all required `__init__.py` files are present
3. Offer to create any missing `__init__.py` files automatically

## Root Cause

The error occurs when Python cannot find the `inner_loop` package. This usually happens due to one of these reasons:

### 1. **Incomplete Directory Copy** (Most Common)
When copying the project to a new machine, the `inner_loop` directory or its subdirectories might not have been copied.

**Required structure:**
```
rl4co git/
├── inner_loop/                    # ← Must exist!
│   ├── __init__.py               # ← Required!
│   └── rl4co/
│       ├── __init__.py           # ← Required!
│       ├── envs/
│       │   ├── __init__.py       # ← Required!
│       │   └── routing/
│       │       ├── __init__.py   # ← Required!
│       │       └── pdptw/
│       │           ├── env.py
│       │           └── sf_generator.py
│       └── models/
│           ├── __init__.py       # ← Required!
│           └── zoo/
│               ├── __init__.py   # ← Required!
│               └── ...
└── ppo_loop/
    ├── meta_train.py
    ├── dvrp_env.py
    └── ...
```

**Fix:**
- Make sure you copied the **entire** `rl4co git` directory, not just `ppo_loop`
- Use `rsync` or `tar` to ensure all files are copied:
  ```bash
  # On source machine (macOS)
  tar -czf rl4co_project.tar.gz "rl4co git/"

  # On destination machine (Linux)
  tar -xzf rl4co_project.tar.gz
  ```

### 2. **Missing `__init__.py` Files**
Some file transfer methods (like certain FTP clients or cloud sync) might skip `__init__.py` files, especially if they're configured to ignore hidden or empty files.

**Check if files exist:**
```bash
cd /path/to/rl4co\ git
find inner_loop -name "__init__.py" -type f
```

**Expected output:** Should list at least these files:
- `inner_loop/__init__.py`
- `inner_loop/rl4co/__init__.py`
- `inner_loop/rl4co/envs/__init__.py`
- `inner_loop/rl4co/envs/routing/__init__.py`
- `inner_loop/rl4co/models/__init__.py`
- `inner_loop/rl4co/models/zoo/__init__.py`

**Fix:**
Run the setup script:
```bash
cd ppo_loop
python setup_package.py
```

### 3. **Python Path Issues**
The code should automatically add the project root to `sys.path`, but if it's not working:

**Manual fix:**
```bash
# Add this to your shell profile (~/.bashrc or ~/.zshrc)
export PYTHONPATH="/path/to/rl4co git:$PYTHONPATH"
```

Or set it before running:
```bash
cd /path/to/rl4co\ git/ppo_loop
PYTHONPATH=/path/to/rl4co\ git python meta_train.py --episodes 1000
```

## Verification Steps

### Step 1: Verify directory structure
```bash
cd /path/to/rl4co\ git
ls -la inner_loop/
ls -la inner_loop/rl4co/
ls -la inner_loop/rl4co/envs/routing/
```

### Step 2: Test the import
```bash
cd /path/to/rl4co\ git
python -c "from inner_loop.rl4co.envs.routing import PDPTWEnv, SFGenerator; print('✓ Import successful')"
```

If this succeeds, the package structure is correct!

### Step 3: Run the training script
```bash
cd ppo_loop
python meta_train.py --episodes 1000
```

## Common Mistakes

❌ **Only copying `ppo_loop` directory**
```
your_machine/
└── ppo_loop/  # ← Missing inner_loop!
```

✅ **Correct: Copy the entire project**
```
your_machine/
└── rl4co git/
    ├── inner_loop/  # ← Required!
    └── ppo_loop/
```

❌ **Using `cp` without `-a` flag (doesn't preserve all files)**
```bash
cp -r "rl4co git" /destination/  # Might miss __init__.py files
```

✅ **Use `tar` or `rsync` for reliable copying**
```bash
tar -czf project.tar.gz "rl4co git/"
# or
rsync -av "rl4co git/" /destination/rl4co\ git/
```

## Still Having Issues?

If the problem persists after following these steps, please provide:

1. **Output of the diagnostic script:**
   ```bash
   cd ppo_loop
   python setup_package.py
   ```

2. **Directory structure:**
   ```bash
   cd /path/to/rl4co\ git
   find . -maxdepth 3 -type d | sort
   ```

3. **Python path information:**
   ```bash
   cd ppo_loop
   python -c "import sys; print('\n'.join(sys.path))"
   ```

4. **Full error traceback** when running `meta_train.py`
