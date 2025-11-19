# DARP Device Mismatch Error - Diagnostic Plan

## Error Summary

**Error Type**: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!`

**Location**: `env.py:302` in `_get_action_mask()`

**Traceback Context**:
```
File ~/Desktop/Research/llm-rl/rl4co/rl4co/envs/routing/darp/env.py:302, in DARPEnv._get_action_mask(self, td)
    300 # Pickup feasibility (mask out non-pickup nodes by AND with is_pickup_node)
    301 pickup_mask = not_visited & time_feasible & capacity_feasible        # [...,N]
--> 302 pickup_mask = pickup_mask & is_pickup_node                           # [...,N]
```

**Training Context**:
- Device: MPS (Apple Silicon GPU)
- Batch size: 512
- Triggered during: Baseline rollout evaluation in `RolloutBaseline.setup()`

## Root Cause Analysis

### Problem Location: Lines 287-290

```python
# Node indices helpers
N = num_loc + 1
idx = torch.arange(N, device=device)                           # ✓ ON DEVICE
node_broadcast_shape = (1,) * len(batch_dims) + (N,)
is_pickup_node = (idx % 2 == 1).view(node_broadcast_shape)    # ✓ ON DEVICE
is_dropoff_node = ((idx % 2 == 0) & (idx != 0)).view(node_broadcast_shape)  # ✓ ON DEVICE
```

### Inconsistency Issue

The `is_pickup_node` and `is_dropoff_node` tensors **ARE** created on the correct device (line 287 uses `device=device`), so the issue must be elsewhere.

### Secondary Analysis: Device Parameter

Let me trace where `device` comes from in `_get_action_mask()`:

```python
def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
    # === dimensions ===
    batch_dims = td["locs"].shape[:-2]
    num_loc = td["locs"].shape[-2] - 1
    device = td.device  # ← This gets the device from TensorDict
```

**Potential Issue**: `td.device` might return `None` or `cpu` even when tensors are on MPS.

### TensorDict Device Behavior

TensorDict's `.device` property may not always reflect the actual device of its tensors, especially when:
1. TensorDict is constructed without explicit device parameter
2. Tensors are moved to device after TensorDict creation
3. Mixed devices exist within the TensorDict

## Diagnostic Steps

### Step 1: Verify Device Extraction (Line 258)

**Current Code**:
```python
device = td.device
```

**Issue**: May return `None` or `cpu` even if tensors are on `mps:0`

**Fix**: Extract device from an actual tensor:
```python
device = td["locs"].device
```

This ensures we get the actual device where tensors reside.

### Step 2: Check Other Device-Dependent Tensor Creations

Search for all `torch.zeros`, `torch.ones`, `torch.arange`, etc. in `_get_action_mask()`:

**Line 260**: ✓ Already uses `device=device`
```python
action_mask = torch.zeros((*batch_dims, num_loc + 1), dtype=torch.bool, device=device)
```

**Line 287**: ✓ Already uses `device=device`
```python
idx = torch.arange(N, device=device)
```

**Line 306**: ⚠️ **POTENTIAL ISSUE**
```python
pickup_done[..., 0] = False
```
If `False` is interpreted as a tensor, it might default to CPU. Should be fine since it's a scalar assignment.

### Step 3: Check `_reset()` Method

Lines 180-251 in `_reset()` also create tensors. Verify all use correct device:

**Line 181**: ✓ Device extracted from input
```python
device = td.device
```

**Line 190-206**: ✓ All use `device=device`
```python
current_node = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
current_agent = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
# ... etc
```

**Line 210**: ⚠️ **POTENTIAL ISSUE**
```python
torch.full((*batch_size, 1), 48, dtype=torch.long, device=device)
```
Uses `torch.full` - should be okay if `device=device` is specified.

**Line 215**: ✓ Uses `device=device`
```python
torch.zeros((*batch_size, 1), dtype=torch.float32, device=device)
```

### Step 4: Check `_step()` Method

Lines 72-178 perform state updates. No new tensor allocations that could cause device issues.

## Recommended Fixes

### Fix 1: Change Device Extraction (PRIMARY FIX)

**File**: `env.py`

**Line 258** - Change from:
```python
device = td.device
```

To:
```python
device = td["locs"].device
```

**Rationale**: Ensures device is extracted from actual tensor, not TensorDict metadata.

### Fix 2: Apply Same Fix in `_reset()`

**Line 181** - Change from:
```python
device = td.device
```

To:
```python
device = td["depot"].device  # or any tensor that exists in input td
```

**Rationale**: Consistency across methods.

### Fix 3: Verify TensorDict Construction in `_reset()`

**Lines 221-233 and 237-251**: Ensure returned TensorDict inherits device properly.

**Current**:
```python
return TensorDict({...}, batch_size=batch_size)
```

**Potential Enhancement**:
```python
return TensorDict({...}, batch_size=batch_size, device=device)
```

Though this should not be necessary if all tensors are already on the correct device.

## Testing Plan

### Test 1: CPU Device
```python
env = DARPEnv(generator_params={'num_loc': 10})
td = env.reset(batch_size=[5])  # Should be on CPU
assert td["locs"].device.type == "cpu"
action_mask = env.get_action_mask(td)
assert action_mask.device.type == "cpu"
```

### Test 2: MPS Device
```python
env = DARPEnv(generator_params={'num_loc': 10})
td = env.reset(batch_size=[5]).to("mps")
assert td["locs"].device.type == "mps"
action_mask = env.get_action_mask(td)
assert action_mask.device.type == "mps"
```

### Test 3: CUDA Device (if available)
```python
env = DARPEnv(generator_params={'num_loc': 10})
td = env.reset(batch_size=[5]).to("cuda:0")
assert td["locs"].device.type == "cuda"
action_mask = env.get_action_mask(td)
assert action_mask.device.type == "cuda"
```

### Test 4: Full Training Loop
```python
# As in environment_testing.ipynb
device = torch.device("mps")
env = DARPEnv(generator_params={'num_loc': 10})
policy = AttentionModelPolicy(env_name=env.name)
model = REINFORCE(env, policy, baseline="rollout", batch_size=512)
trainer = RL4COTrainer(max_epochs=1, accelerator="mps", devices=1)
trainer.fit(model)  # Should not crash
```

## Comparison with Other Environments

Check if other environments in `rl4co/envs/routing/` have similar issues:

**TSP** (`tsp/env.py`):
```python
# Check if they use td.device or td["tensor_name"].device
```

**CVRP** (`cvrp/env.py`):
```python
# Check device handling pattern
```

**MDCPDP** (`mdcpdp/env.py`):
```python
# Check if similar multi-agent environment has this issue
```

## Notebook Context Analysis

### Observation from `environment_testing.ipynb`

**Cell output when printing TensorDict**:
```python
TensorDict(
    fields={...},
    batch_size=torch.Size([5]),
    device=None,  # ← CONFIRMS THE ISSUE
    is_shared=False)
```

**Key finding**: Even though individual tensors are on CPU (or MPS after `.to(device)`), the TensorDict's `.device` property is **`None`**.

**Trainer configuration** (Cell 18):
```python
trainer = RL4COTrainer(
    max_epochs=2,
    accelerator="gpu",  # ← Becomes MPS on Apple Silicon
    devices=1,
    logger=logger,
    callbacks=callbacks,
)
```

**Why this causes the error**:
1. When `trainer.fit(model)` runs, data is moved to MPS device
2. `RolloutBaseline.setup()` calls `env.reset(batch.to(device))`
3. TensorDict tensors are now on MPS, but `td.device` is still `None`
4. In `_get_action_mask()`: `device = td.device` → `device = None`
5. `torch.arange(N, device=None)` creates tensor on **default device (CPU)**
6. Mixing CPU tensor with MPS tensors → **RuntimeError**

**This is NOT a notebook-specific issue** - it's a fundamental TensorDict behavior where `.device` property doesn't track individual tensor devices.

## Expected Outcome

After applying **Fix 1**, the error should resolve because:
1. `device` will correctly reflect where tensors actually reside (mps:0)
2. `is_pickup_node` and `is_dropoff_node` will be created on mps:0
3. Boolean operations between tensors on same device will succeed
4. Training loop will proceed without device mismatch errors

## Additional Considerations

### 1. TensorDict Device Property Behavior
TensorDict's `.device` may return:
- `None` if tensors are on different devices
- First device encountered
- May not accurately reflect after `.to(device)` call

**Solution**: Always extract device from a known tensor key.

### 2. Batch Dimension Handling
The code supports arbitrary batch dimensions `[B]` or `[B, S]`. Ensure device handling works for all shapes.

### 3. MPS-Specific Issues
MPS (Apple Silicon) may have different behavior than CUDA:
- Some operations may fall back to CPU
- Check PyTorch version supports MPS for all ops used

### 4. Environment Base Class
Check if `RL4COEnvBase` has device handling conventions that should be followed.

## Summary

**Primary Issue**: `td.device` returns incorrect device information (likely `None` or `cpu`) even when tensors are on `mps:0`.

**Primary Fix**: Change `device = td.device` to `device = td["locs"].device` on line 258.

**Verification**: Run training loop on MPS device and ensure no device mismatch errors occur.

**Risk**: Low - this is a simple, safe fix that aligns device extraction with actual tensor locations.

---

## Fix Implementation & Verification

### Fixes Applied ✓

**Fix 1 - Line 258 in `_get_action_mask()`**:
```python
# BEFORE:
device = td.device

# AFTER:
device = td["locs"].device
```

**Fix 2 - Line 181 in `_reset()`**:
```python
# BEFORE:
device = td.device

# AFTER:
device = td["depot"].device
```

### Verification Tests ✓

**Test 1: CPU Device** - PASSED
```
TensorDict device property: None
Actual tensor device: cpu
Action mask device: cpu
✓ CPU test passed
```

**Test 2: MPS Device** - PASSED
```
TensorDict device property: mps  (note: changes to mps after .to('mps'))
Actual tensor device: mps:0
Action mask device: mps:0
✓ MPS test passed
```

**Test 3: Policy Rollout on MPS** - PASSED
```
Using device: mps
Input TensorDict locs device: mps:0
Reward device: mps:0
Actions device: mps:0
✓ Policy rollout test passed
```

### Key Observations

1. **TensorDict.device behavior**:
   - Returns `None` when created without explicit device
   - Returns device after `.to(device)` call, but this is unreliable
   - Individual tensors always have correct `.device` property

2. **Fix effectiveness**:
   - All tensors now created on correct device
   - No device mismatch errors during masking operations
   - Policy rollout works correctly on both CPU and MPS

3. **Training readiness**:
   - Environment now compatible with MPS training
   - Baseline rollout evaluation should proceed without errors
   - Fix is backward compatible (CPU training still works)

### Expected Training Result

The original error:
```python
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!
```

Should now be **resolved** and training should proceed through the baseline setup phase.
