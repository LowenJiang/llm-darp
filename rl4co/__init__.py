# Apply compatibility patches for torchrl with tensordict 0.7.2
try:
    from rl4co.utils.torchrl_compat import TORCHRL_AVAILABLE
except ImportError:
    TORCHRL_AVAILABLE = False

from importlib.metadata import version as get_version

# The package version is obtained from the pyproject.toml file
#__version__ = get_version(__package__)


