# Einops: Transpose (Swapping Dimensions)

A common operation in image processing is **transposing**, or swapping dimensions.

Example from exercise (7):

```python
# Input: (c, h, w)
# Output: (c, w, h)
arr7 = einops.rearrange(arr[1], "c h w -> c w h")
```

This works because it performs a **Matrix Transpose** on the spatial dimensions.

### Why it works

Your input `arr[1]` represents a single image with shape `(Channels, Height, Width)`.

- `c`: Color channels (e.g., RGB)
- `h`: Height (Vertical axis / Rows)
- `w`: Width (Horizontal axis / Columns)

When you run `einops.rearrange(..., "c h w -> c w h")`, you are keeping the channels `c` as they are, but **swapping** the position of `h` and `w`.

In Linear Algebra terms, for every channel, you are transposing the matrix:

- A pixel at position `(row, col)` moves to position `(col, row)`.

### Thinking in Coordinates: Where is (0,0)?

It is better to think in **"Matrix Indexing"** because standard Cartesian graphs (like in school math) usually put `(0,0)` at the **bottom-left** with Y pointing up.

In computer images (and matrices):

- **Origin (0,0)** is at the **TOP-LEFT**.
- **Height (h)** corresponds to **Rows** (Y-axis points **DOWN**).
- **Width (w)** corresponds to **Columns** (X-axis points **RIGHT**).

### Visualizing the Transformation

Because the origin is at the top-left:

1. **Original**: The pixel at `(h=10, w=50)` is 10 pixels down, 50 pixels right.
2. **Transposed**: It moves to `(h=50, w=10)`, which is 50 pixels down, 10 pixels right.

This effectively **reflects the image across the main diagonal** (the line connecting the top-left corner to the bottom-right).

**Summary Diagram:**

```text
(0,0) -------------> Width (x)     (0,0) -------------> Width (x)
  |  A . . . . .                     |  A . . D . .
  |  . . . . . .                     |  . . . . . .
  |  . . . . . .         BECOMES     |  . . . . . .
  |  D . . . . .                     |  . . . . . .
  |  . . . . . .                     |  . . . . . .
  v                                  v
Height (y)                         Height (y)

(D is originally at a large Height, small Width.
 After Swap, D is at large Width, small Height).
```
