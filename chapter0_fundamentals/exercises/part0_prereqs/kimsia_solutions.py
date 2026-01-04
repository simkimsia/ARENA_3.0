# %%
# always go https://colab.research.google.com/github/callummcdougall/ARENA_3.0/blob/main/chapter0_fundamentals/exercises/part0_prereqs/0.0_Prerequisites_exercises.ipynb?t=20250910
# in terminal make sure venv activated
# cd chapter0_fundamentals/exercises/part0_prereqs/
# python kimsia_solutions.py

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
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"

# to make life easier for myself
show = True

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
if MAIN and not show:
    print(arr[0].shape)
    display_array_as_img(arr[0])  # plotting the first image in the batch

# %%
if MAIN and not show:
    arr_stacked = einops.rearrange(arr, "b c h w -> c h (b w)")
    print(arr_stacked.shape)
    display_array_as_img(arr_stacked)  # plotting all images, stacked in a row

# %%

"""
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵⚪⚪⚪

You should spend up to ~45 minutes on these exercises collectively.
If you think you get the general idea, then you can skip to the next section.
You shouldn't spend longer than ~10 mins per exercise.
"""
# %%
if MAIN and not show:
    display_soln_array_as_img(1)

# %%
if MAIN and not show:
    # (1) Column Stacking
    # The key to solving this is to run the earlier cells in colab
    # and then at the cell where visually you see both the code and output where
    # 0-5 are laid out hirizontally
    # then the previous cell is askingfor the outcome you need to aim for
    # in this cell
    # then the answer becomes obvious.
    # if need be, revise self-paced/chapter0_fundamentals/einops.md
    # My initial attempt was `arr1 = einops.rearrange(arr, "b c h w -> c w (b h)")``
    # which was wrong as it produced horizontal stacking of images and with the images flipped
    # the realization is that we are reducing a 4D tensor to 3D tensor
    # so the 1st number in a 3D tensor miust still be c => channels
    # and the 2nd is h = height, and soon
    arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")
    display_array_as_img(arr1)

# %%
if MAIN and not show:
    #### (2) Column-stacking and copying
    # In this example we take just the first digit, and copy it along rows using `einops.repeat`.
    display_soln_array_as_img(2)

# %%
if MAIN and not show:
    # Realize that arr is an array of digit images
    # so if we want the first digit, then we can use arr[0]
    # and because we only care about 1 image, so it's 3D tensor, so it's c h w
    # copy along the rors means the height is increased by 2
    arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")
    display_array_as_img(arr2)

# %%
#### (3) Row-stacking and double-copying
# This example is pretty similar to the previous one, except that the part of the original image we need to slice and pass into `einops.repeat` also has a batch dimension of 2 (since it includes the first 2 digits).
if MAIN and not show:
    display_soln_array_as_img(3)

# %%
if MAIN and not show:
    # We wantjust the first 2 of arr since it's 0 and 1
    # sowe use arr[0:2]
    # it's going to be the same batch and channels, but the height is increased by 2
    # Originally i use `einops.repeat(arr[0:2], "b c h w -> b c (b h) w")`
    # but this is wrong as it's going to be 4D tensor
    # the key is to realize taht repeat also can reduce dimensions
    # so we use `einops.repeat(arr[0:2], "b c h w -> c (b h) (2 w)")`
    # i also tried `einops.repeat(arr[0:2], "b c h w -> c (2 h) (2 w)")`
    # but this is wrong as this means left hand side will have b as unexpected identifier
    arr3 = einops.repeat(arr[0:2], "b c h w -> c (b h) (2 w)")
    display_array_as_img(arr3)

# %%
#### (4) Stretching

# The image below was stretched vertically by a factor of 2.
if MAIN and not show:
    display_soln_array_as_img(4)

# %%
if MAIN and not show:
    # again, we only care about 1 image, so it's 3D tensor, so it's arr[0] and c h w on the left
    # stretch vertically means the height is increased by 2
    # initially i didn't think it means repeat because we already use repeat to increase
    # along the rows by using c (2 h) w
    # but i then realized that stretching means we use c (h 2) w instead
    arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w")
    display_array_as_img(arr4)

# %%
#### (5) Split channels

# The image below was created by splitting out the 3 channels of the image (i.e. red, green, blue) and turning these into 3 stacked horizontal images. The output is 2D (the display function interprets this as a monochrome image).
if MAIN and not show:
    display_soln_array_as_img(5)

# %%
if MAIN and not show:
    # we only care about 1 image, so it's 3D tensor, so it's arr[0] and c h w on the left
    # i googled and found https://medium.com/@kyeg/einops-in-30-seconds-377a5f4d641a
    # so i thought the answer is `einops.rearrange(arr[0], "rgb h w -> rgb  h w", rgb=3)`
    # but this is wrong as it's going to be 4D tensor
    # a couple realizations:
    # realization 1:
    # checking the definition of `display_array_as_img` tells us that if
    # parameter input array is a 2D tensor, it will simply grayscale the image.
    # realization 2:
    # we typically think of an image as 2D and this is correct,but when we represent a 2D image as 3D tensor
    # it's better to visualize it as a 3D thing where each channel layer is vertically on top of next layer
    # with RED as top layer, then GREEN, then BLUE. see einops.md#flattening-channels
    # so physically both "c h w" and "h (c w)" are physically 2D images
    # (c w) means we group by channel first and we get red, green, blue looking like 3 "replicas" side by side
    # (w c) means we group by pixels first and we get "stretched" horizontal effect
    arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")
    display_array_as_img(arr5)

# %%
#### (6) Stack into rows & cols

# This requires a rearrange operation with dimensions for row and column stacking.
if MAIN and not show:
    display_soln_array_as_img(6)


# %%
# initially tried `arr6 = einops.rearrange(arr, "b c h w -> c (b h) (3 w)")`
# i got
# einops.EinopsError:  Error while processing rearrange-reduction pattern "b c h w -> c (b h) (3 w)".
# Input tensor shape: (6, 3, 150, 150). Additional info: {}.
# Non-unitary anonymous axes are not supported in rearrange (exception is length 1)
# answer was `arr6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)`

if MAIN and not show:
    arr6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2, b2=3)
    display_array_as_img(arr6)

# %%

#### (7) Transpose

# Here, we've just flipped the model's horizontal and vertical dimensions. Transposing is a fairly common tensor operation.
if MAIN and show:
    display_soln_array_as_img(7)

if MAIN and show:
    # i guessed correctly
    # but i think the key point is to realize the origin 0,0 at
    # top left corner and imagine an imaginary line from
    # top left to bottom right and mirror reflect see @einops-transpose.md
    arr7 = einops.rearrange(arr[1], "c h w -> c w h")
    display_array_as_img(arr7)
