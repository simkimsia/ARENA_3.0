# Understanding `einops.rearrange`

The `rearrange_1` exercise demonstrates how to construct and shape a tensor in one go using `torch.arange` and `einops.rearrange`.

```python
def rearrange_1() -> Tensor:
    """Return the following tensor:
    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    return einops.rearrange(t.arange(3, 9), "(h w) -> h w", h=3, w=2)
```

## Step-by-Step Breakdown

### 1. The Data: `t.arange(3, 9)`

This creates a **1D tensor** starting from `3` up to (but not including) `9`.

* **Data**: `[3, 4, 5, 6, 7, 8]`
* **Shape**: `(6,)`

### 2. The Shape: `einops.rearrange(..., "(h w) -> h w", h=3, w=2)`

This operation reshapes the 1D tensor into a 2D matrix.

* **Pattern `(h w) -> h w`**: This tells `einops` to take a single dimension (of size $h \times w$) and split it into two dimensions: height (`h`) and width (`w`).
* **Sizes `h=3, w=2`**: These define the specific dimensions. Since $3 \times 2 = 6$, it perfectly fits our 6 elements.
* **Filling Order**: `einops` fills the new grid in **row-major order** (filling across the columns first, then moving to the next row).

## Visual Sequence

| Step | Operation | Resulting Data | Shape |
| :--- | :--- | :--- | :--- |
| **Start** | `t.arange(3, 9)` | `[3, 4, 5, 6, 7, 8]` | `(6)` |
| **Reshape** | `h=3, w=2` | `[[3, 4], [5, 6], [7, 8]]` | `(3, 2)` |

The result is a tensor with **3 rows** and **2 columns**, as required.
