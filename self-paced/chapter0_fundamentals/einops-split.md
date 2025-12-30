# Einops: Decomposition (Splitting)

Decomposition is the act of breaking a single dimension into multiple smaller ones.

## 0. Strategy: Shape Accounting

Before writing a complex split, use "Shape Accounting" to solve the puzzle:

1. **Current Shape**: write down the numbers, e.g., `(6, 3, 150, 150)`.
2. **Desired Shape**: write down what you want visually, e.g., `(3, 300, 450)`.
3. **The Budget**: Look at the factors.
   - You need height to go from 150 -> 300 (Factor of **2**).
   - You need width to go from 150 -> 450 (Factor of **3**).
   - Multiply your needed factors: $2 \times 3 = 6$.
4. **The Source**: Look at your input dimensions. Do you have a dimension of size **6** you can "spend"? Yes, the batch!

**Result**: You now know for a fact you need `(2 3)` on the LHS.

---

## 1. RHS Split (The "Unwrap")

This is used when you have a flattened dimension and want to restore its structure.

The rule is: The parentheses are on the **Right** of the arrow.

### Example: Splitting a single row into a grid

Instead of using confusing names like `h` and `w`, let's be explicit about what is happening:

```python
import einops
import numpy as np

# A single row of 6 elements
x = [[1, 2, 3, 4, 5, 6]]  # shape (1, 6)

# We want to split that 6 into 2 rows of 3 elements each
res = einops.rearrange(x,
    "original_row (newrow_count element_per_newrow) -> original_row newrow_count element_per_newrow",
    newrow_count=2,
    element_per_newrow=3
)

# Result shape: (1, 2, 3)
# [[[1, 2, 3],
#   [4, 5, 6]]]
```

**Key Insight**: The parenthesized expression `(newrow_count element_per_newrow)` on the left tells `einops` that the input dimension is a product of these two parts.

---

## 2. LHS Split (The "Prepare")

This is used when you want to treat one dimension (like a batch of 6 images) as a grid (like 2 rows of 3) **before** you start moving them.

The rule is: The parentheses are on the **Left** of the arrow.

### Example: Preparing a batch for a grid display

```python
# arr.shape = (6, 3, 150, 150) -> 6 images
# Goal: Arrange them in 2 rows (b_rows) of 3 images (b_cols)
res = einops.rearrange(arr,
    "(b_rows b_cols) c h w -> c (b_rows h) (b_cols w)",
    b_rows=2
)
```

**How to think about the order**:
In `(b_rows b_cols)`, the **first** name is the "outer" dimension (slowest changing) and the **second** name is the "inner" dimension (fastest changing).

- `b_rows=0` contains images `[0, 1, 2]`
- `b_rows=1` contains images `[3, 4, 5]`

---

## 3. Why no literal numbers like `(2 b2)`?

You might want to write `(2 b2)` to save time, but `einops.rearrange` prevents this for safety:

1. **Symbols only**: Patterns are templates. Integers are treated as "anonymous dimensions."
2. **The "Non-unitary" Error**: Most versions of `einops` forbid non-unitary (size > 1) integers in `rearrange`. This ensures you don't hardcode "magic numbers" into your patterns.
   - Error: `Non-unitary anonymous axes are not supported`.
3. **The Solution**: Always use a name (like `b_rows`) and pass the value as a keyword argument (`b_rows=2`).

---

## 4. Summary: RHS Split vs LHS Split

| Feature | RHS Split (`a -> (b c)`) | LHS Split (`(b c) -> a`) |
| :--- | :--- | :--- |
| **Logic** | "Unwrap" an existing dimension. | "Prepare" a dimension for distribution. |
| **Mental Model** | **Execution**: You are doing the splitting now. | **Preparation**: You are defining the split before you act. |
| **Example** | `arr_flat -> (h w)` | `(b1 b2) h w -> ...` |

---

## 5. Advanced Examples

### Vision Transformer Patches

Splitting height and width into patches:

```python
# Split 4x4 image into 2x2 patches
# (h p1) splits height, (w p2) splits width
einops.rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=2, p2=2)
```

### Transformer Attention Heads

Splitting an embedding dimension:

```python
# Split hidden_dim 6 into 2 heads of 3 dim each
einops.rearrange(x, "b s (heads head_dim) -> b heads s head_dim", heads=2)
```
