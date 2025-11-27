# Complete Linux Setup Instructions

## TL;DR (Quick Start)

On your Linux machine, run these commands:

```bash
cd /home/jiangwolin/"rl4co git"

# Fix the import issues
python fix_imports.py

# Verify setup
cd ppo_loop
python -c "from dvrp_env import DVRPEnv; print('Setup OK!')"

# Start training (use 'cuda' not 'gpu' for GPU!)
python meta_train.py --episodes 12800 --device cuda
```

---

## Detailed Setup Guide

### Prerequisites

1. You have copied the entire `rl4co git` directory to your Linux machine
2. You have a Python environment with all required packages installed

### Step-by-Step Instructions

#### 1. Navigate to the project directory
```bash
cd /home/jiangwolin/"rl4co git"
```

#### 2. Fix import issues
The `inner_loop` package has some files that import `rl4co` instead of `inner_loop.rl4co`. Run the automated fix:

```bash
python fix_imports.py
```

**Expected output:**
```
================================================================================
Import Fixer for inner_loop Package
================================================================================
Searching in: /home/jiangwolin/rl4co git/inner_loop
Found XXX Python files to check

  ✓ Fixed: rl4co/utils/trainer.py
  ✓ Fixed: rl4co/utils/meta_trainer.py
  ✓ Fixed: rl4co/tasks/train.py

================================================================================
✅ Fixed imports in 3 file(s)
================================================================================
```

#### 3. Verify package structure (Optional but recommended)
```bash
cd ppo_loop
python setup_package.py
```

This will check if all required `__init__.py` files exist and offer to create any missing ones.

#### 4. Test the imports
```bash
python -c "from dvrp_env import DVRPEnv; print('✅ Imports working!')"
```

If this succeeds, you're ready to train!

#### 5. Check CUDA availability (for GPU training)
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 6. Start training

**Basic training (CPU):**
```bash
python meta_train.py --episodes 12800
```

**GPU training (⚠️ use `cuda` not `gpu`):**
```bash
python meta_train.py --episodes 12800 --device cuda
```

**With all options:**
```bash
python meta_train.py \
    --episodes 12800 \
    --num-envs 64 \
    --customers 30 \
    --policy-update-interval 3 \
    --save-interval 100 \
    --log-interval 1 \
    --device cuda \
    --seed 42
```

**Resume from checkpoint:**
```bash
python meta_train.py \
    --episodes 25600 \
    --resume ./checkpoints/ppo_agent_ep12800.pt \
    --device cuda
```

---

## Common Errors and Solutions

### ❌ Error 1: `ModuleNotFoundError: No module named 'rl4co'`

**Full error:**
```
File "/home/jiangwolin/rl4co git/inner_loop/rl4co/utils/trainer.py", line 14
    from rl4co import utils
ModuleNotFoundError: No module named 'rl4co'
```

**Cause:** Import paths in `inner_loop` are incorrect.

**Solution:**
```bash
cd /home/jiangwolin/"rl4co git"
python fix_imports.py
```

### ❌ Error 2: `invalid choice: 'gpu'`

**Full error:**
```
error: argument --device: invalid choice: 'gpu' (choose from 'cpu', 'cuda')
```

**Cause:** Used `--device gpu` instead of `--device cuda`.

**Solution:** Use the correct device argument:
```bash
# For GPU training
python meta_train.py --episodes 1000 --device cuda  ✅

# NOT this:
python meta_train.py --episodes 1000 --device gpu   ❌
```

### ❌ Error 3: `ModuleNotFoundError: No module named 'inner_loop'`

**Full error:**
```
ModuleNotFoundError: No module named 'inner_loop'
```

**Cause:** The `inner_loop` directory doesn't exist or isn't in the right place.

**Solution:**
1. Make sure you copied the entire `rl4co git` directory:
   ```bash
   ls -la /home/jiangwolin/"rl4co git"/inner_loop
   ```

2. If missing, re-copy from the source machine using tar:
   ```bash
   # On macOS (source)
   cd ~/Desktop/Research/llm-rl
   tar -czf rl4co_project.tar.gz "rl4co git/"

   # Transfer to Linux, then:
   tar -xzf rl4co_project.tar.gz
   ```

### ❌ Error 4: Missing `__init__.py` files

**Solution:**
```bash
cd /home/jiangwolin/"rl4co git"/ppo_loop
python setup_package.py
# Answer 'yes' when prompted to create missing files
```

---

## Training Script Arguments

```
--episodes              Total training episodes (default: 1000)
--num-envs             Number of parallel environments (default: 64)
--customers            Customers per episode (default: 30)
--save-dir             Checkpoint directory (default: ./checkpoints)
--save-interval        Save every N epochs (default: 100)
--log-interval         Log every N epochs (default: 1)
--policy-update-interval  Update policy every N epochs (default: 1)
--device               Device: 'cpu' or 'cuda' (default: cpu)
--seed                 Random seed (default: 42)
--resume               Path to checkpoint to resume from
--eval                 Run evaluation after training
```

### Example Training Configurations

**Fast testing:**
```bash
python meta_train.py --episodes 640 --num-envs 64
# 10 epochs, finishes in ~10 minutes
```

**Standard training:**
```bash
python meta_train.py \
    --episodes 12800 \
    --num-envs 64 \
    --device cuda \
    --save-interval 100
# 200 epochs, ~2-3 hours on GPU
```

**Long training with less frequent policy updates:**
```bash
python meta_train.py \
    --episodes 25600 \
    --num-envs 64 \
    --policy-update-interval 3 \
    --device cuda \
    --save-interval 50
# 400 epochs, policy updated every 3 epochs
```

---

## Monitoring Training

Training logs are sent to Weights & Biases (wandb). Make sure you're logged in:

```bash
wandb login
```

You can also monitor the console output which shows:
- Average reward
- Average routing cost
- Acceptance rate
- Improvement vs. no negotiation baseline
- Policy/value loss
- Entropy

---

## File Locations

```
/home/jiangwolin/rl4co git/
├── fix_imports.py              # Run this first to fix imports
├── inner_loop/                 # Core RL4CO library (must exist!)
│   ├── __init__.py
│   └── rl4co/
│       ├── envs/
│       └── models/
└── ppo_loop/
    ├── meta_train.py          # Main training script
    ├── ppo_agent.py           # PPO implementation
    ├── dvrp_env.py            # Environment
    ├── setup_package.py       # Verify package structure
    ├── checkpoints/           # Saved models
    └── traveler_decisions_augmented.csv  # Ground truth data
```

---

## Need Help?

1. **Quick fix for most issues:**
   ```bash
   cd /home/jiangwolin/"rl4co git"
   python fix_imports.py
   cd ppo_loop
   python setup_package.py
   ```

2. **Clear Python cache if imports are still failing:**
   ```bash
   find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
   find . -type f -name "*.pyc" -delete 2>/dev/null
   ```

3. **See detailed troubleshooting:**
   - `TROUBLESHOOTING.md` - Comprehensive debugging guide
   - `QUICK_FIX_GUIDE.md` - Quick solutions for common issues
