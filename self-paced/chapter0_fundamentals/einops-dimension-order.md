# Einops: The Golden Rule of Dimension Order

In `einops`, the most important concept to master is the **Order of Dimensions** inside parentheses. This determines how elements are grouped, split, and moved.

## 1. The Right-to-Left Rule (Inner vs. Outer)

Whenever you see dimensions grouped together like `(a b)`, follow this rule:

- **The Right-most name (`b`) is the "Inner" (Fast) dimension.**
- **The Left-most name (`a`) is the "Outer" (Slow) dimension.**

### The "Loop" Analogy

Think of nested for-loops:

```python
for a in range(outer):      # Slow loop
    for b in range(inner):  # Fast loop
        process(array[a][b])
```

- Elements that are **neighbors** in the original array will always fill the `inner` (right-most) dimension first.
- Only after the `inner` dimension is full does the `outer` (left-most) dimension increment.

---

## 2. Examples of Factoring `(a b)`

### The `(3 2)` Example

Imagine you have 6 elements: `[1, 2, 3, 4, 5, 6]`

#### Pattern A: `(row column)` where column=2

- **Logic**: "I want 2 columns." (inner=2)
- **Result**: `[[1, 2], [3, 4], [5, 6]]`
- **Shape**: (3 rows, 2 columns)

#### Pattern B: `(row column)` where column=3

- **Logic**: "I want 3 columns." (inner=3)
- **Result**: `[[1, 2, 3], [4, 5, 6]]`
- **Shape**: (2 rows, 3 columns)

---

## 3. Real World Cases

### Case 1: Image Tiling (Group by Channel)

**Pattern**: `c h w -> h (c w)`

- **Inner**: `w` (width)
- **Outer**: `c` (channels)
- **Logic**: Complete all `w` pixels for the Red channel, then move to the Green channel.
- **Visual**: `[Red Image] [Green Image] [Blue Image]`

### Case 2: Image Stretching (Group by Pixel)

**Pattern**: `c h w -> h (w c)`

- **Inner**: `c` (channels)
- **Outer**: `w` (width)
- **Logic**: Move to a pixel, output all 3 channels (RGB), then move to the next pixel.
- **Visual**: `[r1, g1, b1, r2, g2, b2, ...]`

---

## 4. Scaling Up: Multi-Factor Grouping `(a b c)`

Can you go beyond two dimensions? **Absolutely.** The same exact logic applies recursively from right-to-left.

### The 3D Loop Analogy

Grouping three factors `(a b c)` is like three nested for-loops:

```python
for a in range(outer):
    for b in range(mid):
        for c in range(inner): # Fastest loop (neighboring elements)
            process(array[a][b][c])
```

1. **`c`** (Right-most) moves every 1 step (Neighbors).
2. **`b`** (Middle) moves only after `c` completes a full cycle.
3. **`a`** (Left-most) moves only after `b` completes a full cycle.

### Concrete Example: 12 elements

Imagine you have 12 elements: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`

**Pattern**: `(page row col)` where `row=2, col=3`

- `col` (inner=3) groups elements into 3: `[1, 2, 3]`, `[4, 5, 6]`, ...
- `row` (mid=2) groups those into sets of 2: `[[1, 2, 3], [4, 5, 6]]`, ...
- `page` (outer=2) groups everything into 2:

```text
Page 1: [[1,  2,  3], [4,  5,  6]]
Page 2: [[7,  8,  9], [10, 11, 12]]
```

---

## 5. The "Missing Factor" in Reductions

When using `einops.reduce` for operations like **Max Pooling**, the order tells you *what* you are pooling.

**Example**: `(h 2)` on the LHS.

- `2` is on the right (**Inner**).
- This means you are grouping **adjacent** pixels.
- If the `2` is missing from the RHS, those adjacent pixels are merged!

---

## Summary: When in Doubt
If you want to group **adjacent** things together (like pixels in a 2x2 pool or a batch into a grid), the factor (the size of the group) should almost always be on the **RIGHT**.

> [!IMPORTANT]
> **Right = Inner = Fast = Neighbors**
