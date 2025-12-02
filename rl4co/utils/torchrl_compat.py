"""
Compatibility shim for torchrl with tensordict 0.7.2
"""

# Patch for missing MemmapTensor in tensordict 0.7.2
try:
    from tensordict.memmap import MemmapTensor
except ImportError:
    # Create a dummy MemmapTensor for compatibility
    import torch
    from tensordict import TensorDict

    class MemmapTensor(torch.Tensor):
        """Dummy MemmapTensor for compatibility with older tensordict versions"""
        def __new__(cls, *args, **kwargs):
            # Just create a regular tensor
            return super().__new__(cls)

    # Monkey patch it into tensordict.memmap
    import tensordict.memmap
    tensordict.memmap.MemmapTensor = MemmapTensor

    # Also try MemoryMappedTensor if needed
    try:
        from tensordict.memmap import MemoryMappedTensor
    except ImportError:
        class MemoryMappedTensor(torch.Tensor):
            """Dummy MemoryMappedTensor for compatibility"""
            def __new__(cls, *args, **kwargs):
                return super().__new__(cls)
        tensordict.memmap.MemoryMappedTensor = MemoryMappedTensor

# Now try to import torchrl
try:
    import torchrl
    TORCHRL_AVAILABLE = True
except Exception as e:
    TORCHRL_AVAILABLE = False
    import warnings
    warnings.warn(f"torchrl not available or incompatible: {e}")
