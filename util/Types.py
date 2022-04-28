from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

"""
Custom class that redefines various types to increase clarity.
"""
Key = Union[str, int]  # for dictionaries, we usually use strings or ints as keys
ConfigDict = Dict[Key, Any]  # A (potentially nested) dictionary containing the "params" section of the .yaml file
RecordingDict = Dict[Key, Any]  # Alias for recording dicts
EntityDict = Dict[Key, Union[Dict, str]]  # potentially nested dictionary of entities
ScalarDict = Dict[Key, float]  # dict of scalar values
ValueDict = Dict[Key, Any]
Result = Union[List, int, float, np.ndarray]
