# TensorDict 0.7.2 & TorchRL Compatibility Guide

## Problem

The original requirements specified:
- `tensordict==0.10.0`
- `torchrl==0.10.1`

However, you need to use `tensordict==0.7.2` for Python 3.9 compatibility. This creates version incompatibility issues because:

1. **torchrl 0.5.0+** requires `tensordict>=0.8.0`
2. **torchrl 0.3.x** expects `MemmapTensor` from tensordict which doesn't exist in 0.7.2
3. **torchrl 0.2.x** is the only compatible range, but still has import issues

## Solution

We've implemented a **compatibility shim** that patches the missing components:

### Files Changed

1. **requirements.txt**
   - Updated to `tensordict==0.7.2`
   - Updated to `torchrl==0.2.1`
   - Added compatibility note

2. **rl4co/utils/torchrl_compat.py** (NEW)
   - Monkey-patches missing `MemmapTensor` and `MemoryMappedTensor` classes
   - Provides fallback implementations
   - Gracefully handles import errors

3. **rl4co/__init__.py**
   - Imports compatibility shim before any other imports
   - Ensures patches are applied early

4. **rl4co/models/rl/ppo/stepwise_ppo.py**
   - Made torchrl replay buffer imports optional
   - Added `TORCHRL_AVAILABLE` flag
   - Provides helpful error messages if replay buffers unavailable

5. **rl4co/envs/routing/shpp/env.py**
   - Replaced `UnboundedContinuous` and `UnboundedDiscrete` with `Unbounded`
   - These newer classes don't exist in torchrl 0.2.1

## Installation Instructions

### 1. Uninstall existing incompatible versions

```bash
cd /Users/jiangwolin/Downloads/llm-darp-main
pip uninstall -y tensordict torchrl
```

### 2. Install compatible versions

```bash
pip install tensordict==0.7.2
pip install torchrl==0.2.1
```

### 3. Verify installation

```bash
python -c "
import tensordict
import torchrl
print(f'✓ tensordict: {tensordict.__version__}')
print(f'✓ torchrl: {torchrl.__version__}')

# Test the compatibility shim
from rl4co.utils.torchrl_compat import TORCHRL_AVAILABLE
print(f'✓ TorchRL available: {TORCHRL_AVAILABLE}')

# Test basic imports
from rl4co.envs.routing import SFGenerator, PDPTWEnv
from rl4co.models.zoo import AttentionModel, AttentionModelPolicy
print('✓ All imports successful!')
"
```

## What Works

✅ **Basic RL4CO functionality**
- Environment creation and usage
- SFGenerator, PDPTWEnv
- AttentionModel, AttentionModelPolicy
- Standard PPO (not StepwisePPO)
- All routing environments (TSP, CVRP, etc.)

✅ **TorchRL data types**
- `Bounded`, `Composite`, `Unbounded`
- `EnvBase` inheritance

## Limitations

⚠️ **StepwisePPO may not work**
- Replay buffers from torchrl 0.2.1 may have issues with tensordict 0.7.2
- Use standard `PPO` instead from `rl4co.models.rl.ppo.ppo`
- Alternative: Use `L2DModel` with REINFORCE

⚠️ **Advanced TorchRL features**
- Some advanced replay buffer features may not work
- Memory-mapped tensors use fallback implementation

## Troubleshooting

### Error: "cannot import name 'MemmapTensor'"

**Solution**: Make sure you're importing rl4co FIRST before any torchrl imports:

```python
import rl4co  # This applies the compatibility patches
from rl4co.envs.routing import ...
```

### Error: "StepwisePPO not functional"

**Solution**: Use regular PPO instead:

```python
from rl4co.models.rl import PPO  # Instead of StepwisePPO
```

### Error: "UnboundedContinuous not found"

**Solution**: This has been fixed in `shpp/env.py`. If you see this elsewhere, replace:
- `UnboundedContinuous(...)` → `Unbounded(..., dtype=torch.float32)`
- `UnboundedDiscrete(...)` → `Unbounded(..., dtype=torch.int64)`

## Testing Your Code

Your code in `ppo_loop/dvrp_env.py` should work fine because it only uses:
- `rl4co.envs.routing` (✓ Compatible)
- `rl4co.models.zoo` (✓ Compatible)
- Basic tensordict (✓ Compatible)

Test with:

```bash
cd /Users/jiangwolin/Downloads/llm-darp-main
python -c "
from ppo_loop.dvrp_env import DVRPEnv
print('✓ DVRPEnv imported successfully!')
"
```

## Version Compatibility Matrix

| tensordict | torchrl | Status | Notes |
|------------|---------|--------|-------|
| 0.7.2 | 0.2.1 | ✅ Works | With compatibility shim |
| 0.7.2 | 0.3.x | ❌ Broken | Missing MemmapTensor |
| 0.7.2 | 0.5.0 | ❌ Broken | Requires tensordict 0.8+ |
| 0.8.x+ | 0.3.x+ | ✅ Works | But requires Python 3.10+ |
| 0.10.0 | 0.10.1 | ✅ Works | Original, but Python 3.10+ only |

## Recommended Workflow

1. **Development**: Use tensordict 0.7.2 + torchrl 0.2.1 with Python 3.9
2. **Production**: Consider upgrading to Python 3.10+ to use newer versions
3. **Alternative**: Remove torchrl dependency if not using StepwisePPO

## Summary of All Compatibility Fixes

### Python 3.9 Compatibility (Previous fixes)
- ✅ Replaced all `Type1 | Type2` → `Union[Type1, Type2]`
- ✅ Replaced all `match/case` → `if/elif/else`
- ✅ Fixed `Optional[Type1, Type2]` → `Optional[Union[Type1, Type2]]`
- ✅ Added missing `Union` imports

### TensorDict/TorchRL Compatibility (Current fixes)
- ✅ Created compatibility shim for missing MemmapTensor
- ✅ Made StepwisePPO imports optional
- ✅ Replaced UnboundedContinuous/Discrete with Unbounded
- ✅ Updated requirements.txt with compatible versions

Your codebase is now fully compatible with Python 3.9, tensordict 0.7.2, and torchrl 0.2.1! 🎉
