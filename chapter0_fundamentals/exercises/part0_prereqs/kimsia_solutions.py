# %%

# import math
# import os
import sys
from pathlib import Path

import einops
import numpy as np

# import torch as t
# from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

# import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img

MAIN = __name__ == "__main__"

# %%

if MAIN:
    arr = np.load(section_dir / "numbers.npy")

# %%

"""
arr is a 4D numpy array.

The first axes corresponds to the number, and the next three axes are channels (i.e. RGB), height and width respectively.
You have the function utils.display_array_as_img
which takes in a numpy array and displays it as an image.

There are two possible ways this function can be run:

If the input is three-dimensional, the dimensions are interpreted as (channel, height, width) - in other words, as an RGB image.
If the input is two-dimensional, the dimensions are interpreted as (height, width) - i.e. a monochrome image.
For example:
"""

# %%
if MAIN:
    print(arr[0].shape)
    display_array_as_img(arr[0])  # plotting the first image in the batch

# %%
if MAIN:
    arr_stacked = einops.rearrange(arr, "b c h w -> c h (b w)")
    print(arr_stacked.shape)
    display_array_as_img(arr_stacked)  # plotting all images, stacked in a row
