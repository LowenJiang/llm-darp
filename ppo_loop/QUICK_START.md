# Quick Start Guide

## Running the Training

The code is now ready to run with the `rl4co` package registered instead of `inner_loop`.

### Quick Test (Recommended First)

Test that all imports work correctly:

```bash
cd "/home/jiangwolin/rl4co git/ppo_loop"
export PYTHONPATH="/home/jiangwolin/rl4co git:$PYTHONPATH"
python3 test_imports.py
```

### Run Training

```bash
cd "/home/jiangwolin/rl4co git/ppo_loop"
export PYTHONPATH="/home/jiangwolin/rl4co git:$PYTHONPATH"
python3 meta_train.py --episodes 1000 --device cuda
```

**Or use the provided shell script:**

```bash
cd "/home/jiangwolin/rl4co git/ppo_loop"
chmod +x run_meta_train.sh
./run_meta_train.sh --episodes 1000 --device cuda
```

## What Was Changed

### 1. Package Imports (100+ files)
- Changed `from inner_loop.rl4co` → `from rl4co` throughout the rl4co directory

### 2. CUDA Support Added
- ✓ Oracle model (AttentionModel) now loads on specified device
- ✓ All tensor operations moved to correct device
- ✓ Batched neural oracle calls use GPU when available

### 3. Files Modified

| File | Changes |
|------|---------|
| `dvrp_env.py` | Added `device` parameter, model loads on device, tensors moved to device |
| `vectorized_env.py` | Batched solver moves tensors to device |
| `meta_train.py` | Passes `device` parameter to environments |
| `rl4co/**/*.py` | Import paths updated from `inner_loop.rl4co` to `rl4co` |

## Training Options

```bash
python3 meta_train.py \
    --episodes 1000 \        # Total training episodes
    --device cuda \          # Use 'cuda' for GPU, 'cpu' for CPU
    --num-envs 64 \         # Number of parallel environments
    --customers 30 \         # Customers per episode
    --save-interval 100 \    # Save checkpoint every N epochs
    --log-interval 1 \       # Log stats every N epochs
    --seed 42               # Random seed
```

## GPU Memory Management

If you encounter CUDA out of memory errors, reduce parallel environments:

```bash
python3 meta_train.py --episodes 1000 --device cuda --num-envs 32
```

Or use CPU:

```bash
python3 meta_train.py --episodes 1000 --device cpu
```

## Troubleshooting

### "No module named 'torch'"
Install required packages (PyTorch, numpy, pandas, gymnasium, tensordict, wandb, ortools)

### "No module named 'rl4co'"
Set PYTHONPATH:
```bash
export PYTHONPATH="/home/jiangwolin/rl4co git:$PYTHONPATH"
```

### Check the detailed documentation
See [SETUP_SUMMARY.md](SETUP_SUMMARY.md) for complete details.
