# Setup Summary - RL4CO Package Registration

## Changes Made

This document summarizes all changes made to make the codebase runnable with `rl4co` package instead of `inner_loop`.

### 1. Import Path Fixes

**File: `/home/jiangwolin/rl4co git/rl4co/**/*.py`**
- Replaced all `from inner_loop.rl4co` with `from rl4co`
- Replaced all `import inner_loop.rl4co` with `import rl4co`
- Total files affected: ~100+ files in the rl4co directory

**Command used:**
```bash
find rl4co -type f -name "*.py" -exec sed -i 's/from inner_loop\.rl4co/from rl4co/g' {} +
find rl4co -type f -name "*.py" -exec sed -i 's/import inner_loop\.rl4co/import rl4co/g' {} +
```

### 2. Python Path Configuration

**File: `ppo_loop/dvrp_env.py`**
- Added sys.path modification to import rl4co from parent directory:
```python
# Add parent directory to sys.path to import rl4co
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### 3. CUDA/Device Support

#### File: `ppo_loop/dvrp_env.py`
**Changes:**
1. Added `device` parameter to `__init__`:
   ```python
   def __init__(self, ..., device: str = 'cpu'):
       self.device = device
   ```

2. Updated model loading to use the device (with CPU loading to avoid RNG state issues):
   ```python
   # Always load to CPU first to avoid RNG state errors
   ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
   # Then move model to desired device
   self.policy = model.policy.to(self.device)
   self.policy.eval()  # Set to evaluation mode
   ```

3. Updated policy forward pass to move tensors to device:
   ```python
   current_requests_on_device = self.current_requests.to(self.device)
   out = self.policy(current_requests_on_device, phase='test', ...)
   ```

#### File: `ppo_loop/vectorized_env.py`
**Changes:**
1. Updated `_batch_solve_routing` to move batched tensors to device:
   ```python
   device = next(self.policy.parameters()).device
   batched_td = batched_td.to(device)
   ```

#### File: `ppo_loop/meta_train.py`
**Changes:**
1. Added `device` parameter to both VectorizedDVRPEnv instances:
   ```python
   vec_env = VectorizedDVRPEnv(..., device=device)
   baseline_vec_env = VectorizedDVRPEnv(..., device=device)
   ```

### 4. Run Script Created

**File: `ppo_loop/run_meta_train.sh`**
- Bash script that sets PYTHONPATH and runs meta_train.py
- Usage: `./run_meta_train.sh --episodes 1000 --device cuda`

## How to Run

### Method 1: Using Python directly with PYTHONPATH

```bash
cd "/home/jiangwolin/rl4co git/ppo_loop"
export PYTHONPATH="/home/jiangwolin/rl4co git:$PYTHONPATH"
python3 meta_train.py --episodes 1000 --device cuda
```

### Method 2: Using the run script

```bash
cd "/home/jiangwolin/rl4co git/ppo_loop"
chmod +x run_meta_train.sh
./run_meta_train.sh --episodes 1000 --device cuda
```

## Verification Checklist

- [x] All `inner_loop.rl4co` imports replaced with `rl4co`
- [x] sys.path configuration added to import rl4co from parent directory
- [x] Device parameter added to DVRPEnv
- [x] Model loading uses correct device
- [x] Policy forward pass moves tensors to correct device
- [x] Batched solver moves tensors to correct device
- [x] meta_train.py passes device to environments

## Expected Behavior

When running with `--device cuda`:
1. The oracle model (AttentionModel) loads on CUDA
2. All TensorDict inputs are moved to CUDA before forward pass
3. PPO agent trains on CUDA (as specified in agent initialization)
4. Training should utilize GPU acceleration

## Troubleshooting

### Import Error: No module named 'torch'
You need to install PyTorch and other dependencies. The code expects:
- torch
- numpy
- pandas
- gymnasium
- tensordict
- wandb
- ortools

### Import Error: No module named 'rl4co'
Make sure to set PYTHONPATH or use the run script which sets it automatically.

### CUDA out of memory
Try reducing `--num-envs` parameter (default is 64):
```bash
python3 meta_train.py --episodes 1000 --device cuda --num-envs 32
```
