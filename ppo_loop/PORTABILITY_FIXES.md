# Portability Fixes for Cross-Device Compatibility

## Problem
The codebase had hardcoded absolute paths specific to macOS, causing `ModuleNotFoundError` when running on different machines (e.g., Linux).

## Changes Made

### 1. **dvrp_env.py**
- **Line 16**: Replaced hardcoded `sys.path.append("/Users/jiangwolin/...")` with dynamic path resolution:
  ```python
  PROJECT_ROOT = Path(__file__).resolve().parent.parent
  if str(PROJECT_ROOT) not in sys.path:
      sys.path.insert(0, str(PROJECT_ROOT))
  ```
- **Line 87**: Changed default `model_path` from hardcoded path to `None`
- **Line 708**: Updated test function to use relative path:
  ```python
  decisions_path = Path(__file__).parent / "traveler_decisions_augmented.csv"
  ```

### 2. **meta_train.py**
- **Line 164**: Changed from hardcoded path to relative path:
  ```python
  traveler_decisions_path = Path(__file__).parent / "traveler_decisions_augmented.csv"
  ```

### 3. **embedding.py**
- **Added**: Import `from pathlib import Path`
- **Line 73**: Changed default parameter from hardcoded path to `None`
- **Lines 86-87**: Added default path resolution:
  ```python
  if csv_path is None:
      csv_path = Path(__file__).parent / "traveler_decisions_augmented.csv"
  ```

### 4. **vectorized_env.py**
- **Added**: Import `from pathlib import Path`
- **Line 281**: Updated test function to use relative path:
  ```python
  decisions_path = Path(__file__).parent / "traveler_decisions_augmented.csv"
  ```

## Benefits
- ✅ Code now works on any operating system (macOS, Linux, Windows)
- ✅ No need to modify paths when moving to different machines
- ✅ Paths are automatically resolved relative to the script location
- ✅ No hardcoded machine-specific paths

## Usage
Simply copy the entire `rl4co git` directory to any machine and run:
```bash
cd "path/to/rl4co git/ppo_loop"
python meta_train.py --episodes 12800
```

The code will automatically resolve all paths correctly regardless of the directory location.
