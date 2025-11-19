# DARP Device Mismatch Fix - Summary

## Problem
Training the DARP environment on MPS (Apple Silicon GPU) failed with:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!
```

## Root Cause
TensorDict's `.device` property returns `None` (or unreliable values) even when individual tensors are on specific devices like MPS. This caused:
- `device = td.device` → `device = None`
- `torch.arange(N, device=None)` → creates tensor on CPU
- CPU tensor mixed with MPS tensors → RuntimeError

## Fixes Applied

### File: `rl4co/envs/routing/darp/env.py`

**Change 1 - Line 258** (in `_get_action_mask()`):
```python
# Before:
device = td.device

# After:
device = td["locs"].device
```

**Change 2 - Line 181** (in `_reset()`):
```python
# Before:
device = td.device

# After:
device = td["depot"].device
```

## Verification

All tests passed ✓:
- **CPU device**: Action masks created correctly on CPU
- **MPS device**: Action masks created correctly on MPS
- **Policy rollout**: Complete workflow works on MPS without errors

## Impact
- ✓ Training now works on MPS (Apple Silicon GPU)
- ✓ Backward compatible with CPU training
- ✓ Should also work on CUDA devices
- ✓ No changes to environment logic or behavior
- ✓ Simple, safe fix that extracts device from actual tensors

## Next Steps
You can now run the training in `examples/environment_testing.ipynb` without the device mismatch error. The baseline rollout evaluation should complete successfully.
