import numpy as np

## Basic NumPy Operations

# Convert list to NumPy array
lst = [1, 2, 3, 4, 5]
arr = np.array(lst)
print("Array from list:", arr)

# Array from 1 to 10
arr_1_10 = np.arange(1, 11)
print("1 to 10 array:", arr_1_10)

# Maximum value
arr_max = np.array([10, 25, 5, 40, 15])
print("Maximum value:", np.max(arr_max))

# Identity matrix
identity = np.eye(3)
print("Identity Matrix:\n", identity)

# Mean
print("Mean:", np.mean(arr_max))

# Random 5x5 integers
rand_int = np.random.randint(1, 101, (5, 5))
print("Random 5x5 integers:\n", rand_int)

# Concatenate arrays horizontally
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("Concatenated:", np.hstack((a, b)))

# Reshape to 2D
arr_reshape = np.arange(6)
print("Reshaped array:\n", arr_reshape.reshape(3, -1))


# Intermediate Operations

# Dot product
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print("Dot product:", np.dot(x, y))

# Non-zero indices
arr_nz = np.array([0, 1, 2, 0, 3])
print("Non-zero indices:", np.nonzero(arr_nz))

# First 10 even numbers
even = np.arange(2, 21, 2)
print("Even numbers:", even)

# Sort descending
print("Descending sort:", np.sort(arr_max)[::-1])

# Standard deviation
print("Standard deviation:", np.std(arr_max))

# Random floats
rand_float = np.random.rand(5, 5)
print("Random floats:\n", rand_float)

# Normalize
norm = (arr_max - np.min(arr_max)) / (np.max(arr_max) - np.min(arr_max))
print("Normalized:", norm)


# Advanced Operations

# Unique elements
arr_unique = np.array([1, 2, 2, 3, 4, 4])
print("Unique:", np.unique(arr_unique))

# Cumulative sum (rows)
matrix = np.array([[1, 2], [3, 4]])
print("Cumulative sum:\n", np.cumsum(matrix, axis=1))

# Linspace
lin = np.linspace(0, 1, 10)
print("Linspace:", lin)

# Square root
print("Square root:", np.sqrt(arr_max))

# Random integers (-10 to 10)
rand_int2 = np.random.randint(-10, 11, (3, 3))
print("Random -10 to 10:\n", rand_int2)


# Statistical & Matrix Operations

# Median
print("Median:", np.median(arr_max))

# Reshape 4x4 to 2x8
mat_4x4 = np.arange(1, 17).reshape(4, 4)
print("Reshaped 2x8:\n", mat_4x4.reshape(2, 8))

# Absolute difference
a1 = np.array([1, 2, 3])
a2 = np.array([3, 2, 1])
print("Absolute difference:", np.abs(a1 - a2))

# Replace negatives
rand_arr = np.random.randint(-5, 5, (3, 3))
rand_arr[rand_arr < 0] = 0
print("Negatives replaced:\n", rand_arr)

# Cumulative product (columns)
print("Cumulative product:\n", np.cumprod(matrix, axis=0))

# Fibonacci
fib = [0, 1]
for i in range(2, 10):
    fib.append(fib[-1] + fib[-2])
print("Fibonacci:", np.array(fib))


# Final Computations
# ================================

# Exponential
print("Exponential:", np.exp(arr_max))

# Common elements
print("Common elements:", np.intersect1d(a1, a2))

# Max index
rand_5x5 = np.random.randint(1, 100, (5, 5))
print("Max index:", np.unravel_index(np.argmax(rand_5x5), rand_5x5.shape))

# Pearson correlation
p = np.array([1, 2, 3])
q = np.array([4, 5, 6])
print("Pearson correlation:", np.corrcoef(p, q)[0, 1])

# Diagonal sum
matrix_diag = np.array([[1, 2], [3, 4]])
print("Diagonal sum:", np.trace(matrix_diag))
