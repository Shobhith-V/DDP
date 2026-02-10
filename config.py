"""
Global configuration and hyperparameters for the brain–heart feedback project.
"""

# Mixed-precision and logging/debug options
use_half_precision: bool = False
debug_interval: int = 100

# Default data paths (can be overridden when calling the helpers)
DEFAULT_ECG_FIF_PATH = "transdef_mf2pt2_rest_raw_309.fif"
DEFAULT_EEG_SCOUT_MAT_PATH = "scout_id_309.mat"

# NOTE: The structural connectivity matrix file used in the original notebook
# was named something like "SC_CC120309-27.mat". Adjust this path to match
# your local file if it differs.
DEFAULT_SC_MAT_PATH = "SC_CC120309-27.mat"

