# Broadcasting

Broadcasting is a powerful mechanism in NumPy and PyTorch that allows elementwise operations on tensors of different shapes. When two tensors are involved in an elementwise operation, NumPy/PyTorch tries to broadcast them (i.e., copy them along dimensions) so that they both have the same shape.

## The Rules of Broadcasting

The rules of broadcasting are as follows:

1. **Prepend dummy dimensions**: You can prepend dummy dimensions (of size 1) to the start of a tensor until both have the same number of dimensions.
2. **Repeat dimensions**: After this point, if some dimension has size 1 in one of the tensors, it can be repeated until it matches the size of the corresponding dimension in the other tensor.

### Example: Adding a vector to a batch

Suppose we have a 2D batch of data with shape `(N, k)` (i.e., `N` separate datapoints, each being a vector of length `k`). Suppose we want to add a vector `vec` of length `k` to each datapoint. This is a valid operation:

1. `vec` gets prepended with a dummy dimension so it has shape `(1, k)`, and both are 2D.
2. `vec` gets repeated along the first dimension so it has shape `(N, k)`, matching the shape of `data`.

The output has shape `(N, k)`, and elements `output[i, j] = data[i, j] + vec[j]`.

---

## Warm-up Exercises

Can you figure out which of the following are valid, and which will raise errors?

### Exercise 1

```python
x = t.ones((3, 1, 5))
y = t.ones((1, 4, 5))

z = x + y
```

<details>
<summary>Answer</summary>

This is **valid**. The 0th dimension of `y` and the 1st dimension of `x` can both be copied so that `x` and `y` have the same shape: `(3, 4, 5)`. The resulting array `z` will also have shape `(3, 4, 5)`.

This illustrates an important point: it's not always the case that one tensor is strictly smaller and the other is strictly bigger. Sometimes, both tensors will get expanded.
</details>

### Exercise 2

```python
x = t.ones((8, 2, 6))
y = t.ones((8, 2))

z = x + y
```

<details>
<summary>Answer</summary>

This is **not valid**. We first need to expand `y` by prepending a dimension to the front, making it `(1, 8, 2)`. The last two dimensions of `x` are `(2, 6)`, which won't broadcast with `y`'s `(8, 2)`.
</details>

### Exercise 3

```python
x = t.ones((8, 2, 6))
y = t.ones((2, 6))

z = x + y
```

<details>
<summary>Answer</summary>

This is **valid**. Once PyTorch expands `y` by prepending a single dimension to the front, it becomes `(1, 2, 6)`, which can then be broadcast with `x`.
</details>

### Exercise 4

```python
x = t.ones((10, 20, 30))
y = t.ones((20, 1))

z = x + y
```

<details>
<summary>Answer</summary>

This is **valid**. Once PyTorch expands `y` by prepending a single dimension to the front, it becomes `(1, 20, 1)`, which can then be broadcast with `x` (copying along the first and last dimensions).
</details>

### Exercise 5

```python
x = t.ones((4, 1))
y = t.ones((4,))

z = x + y
```

<details>
<summary>Answer</summary>

This is **valid**. NumPy/PyTorch will expand `y` to `(1, 4)`, then broadcast both to `(4, 4)`.

> [!WARNING]
> Although this won't raise an error, it's very possible that this isn't what was intended. A common source of mistakes is when you add 2 tensors thinking they're the same shape, but one actually has a dummy dimension you forgot about. You'll just have to be vigilant (e.g., adding `assert` statements).

---

## Reshaping for Broadcasting

Einops is a useful tool for reshaping tensors to enable broadcasting. However, if you just need to add or remove a dummy dimension, you can use built-in PyTorch methods:

- `tensor.unsqueeze(dim)`: Returns a new tensor with a dummy dimension of size 1 inserted at position `dim`.
- `tensor.squeeze(dim)`: Returns a tensor with the dimension at position `dim` removed (if it had size 1).

### Examples

```python
x = t.ones((3, 1, 5))

print(x.unsqueeze(3).shape)  # (3, 1, 5, 1) - dummy dimension added at end
print(x.squeeze(1).shape)    # (3, 5)       - dimension at index 1 removed
print(x.squeeze(0).shape)    # (3, 1, 5)    - no change (dimension 0 has size 3)
```

---

> [!TIP]
> For more advanced reshaping and to see how to construct tensors from scratch using these tools, check out the [Einops Rearrange Breakdown](file:///Users/kimsia/Projects/ARENA_3.0/chapter0_fundamentals/exercises/part0_prereqs/einops_rearrange.md).
