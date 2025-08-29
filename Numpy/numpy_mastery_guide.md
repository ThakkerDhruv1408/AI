# The Ultimate NumPy Mastery Guide for AI/ML Engineers

*The 20% of NumPy that powers 80% of your AI/ML work*

---

## Table of Contents

1. [Mathematical Foundation: The Language of AI/ML](#mathematical-foundation)
2. [NumPy Fundamentals: Your New Superpower](#numpy-fundamentals)
3. [Array Creation & Manipulation: Building Your Data](#array-creation)
4. [Indexing & Slicing: Accessing Your Universe](#indexing-slicing)
5. [Mathematical Operations: The Heart of AI/ML](#mathematical-operations)
6. [Broadcasting: NumPy's Secret Weapon](#broadcasting)
7. [Linear Algebra: The AI/ML Powerhouse](#linear-algebra)
8. [Statistical Operations: Understanding Your Data](#statistical-operations)

---

## 1. Mathematical Foundation: The Language of AI/ML {#mathematical-foundation}

### Why This Matters for AI/ML
Before diving into NumPy, you need to understand that **every AI/ML algorithm is fundamentally about manipulating matrices and vectors**. Neural networks? Matrix multiplications. Image processing? 2D/3D arrays. Data preprocessing? Vector operations. NumPy is your mathematical translator.

### Essential Math Concepts

#### Scalars, Vectors, and Matrices
```
Scalar: Just a number → 5
Vector: A 1D array → [1, 2, 3, 4]
Matrix: A 2D array → [[1, 2], [3, 4]]
Tensor: 3D+ array → [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
```

#### Visual Intuition
- **Vector**: Think of coordinates (x, y, z) or feature values [height, weight, age]
- **Matrix**: Think of a spreadsheet or an image (rows × columns)
- **Tensor**: Think of a stack of images or a video (frames × height × width)

#### Key Operations You'll Use Daily
1. **Dot Product**: Measures similarity between vectors
2. **Matrix Multiplication**: Core of neural networks
3. **Element-wise Operations**: Feature scaling, activations
4. **Reshaping**: Preparing data for different layers

### Practice Exercise 1: Mathematical Intuition
**Challenge**: Before writing any code, answer these:
1. If you have 1000 images of 28×28 pixels, what shape tensor would you need?
2. To multiply two matrices A(3×4) and B, what must B's shape be?
3. What's the difference between element-wise multiplication and matrix multiplication?

**Answers**: 
1. (1000, 28, 28) or (1000, 784) if flattened
2. (4, n) where n can be any number
3. Element-wise: [1,2] × [3,4] = [3,8]; Matrix: different operation entirely

---

## 2. NumPy Fundamentals: Your New Superpower {#numpy-fundamentals}

### Why NumPy Dominates AI/ML
- **Speed**: 10-100x faster than pure Python
- **Memory Efficient**: Stores data in contiguous memory blocks
- **Broadcasting**: Performs operations on different shaped arrays automatically
- **Ecosystem**: Foundation for pandas, scikit-learn, TensorFlow, PyTorch

### The NumPy Array (ndarray): Your Primary Tool

```python
import numpy as np

# The basic building block
arr = np.array([1, 2, 3, 4])
print(f"Shape: {arr.shape}")      # (4,)
print(f"Data type: {arr.dtype}")  # int64
print(f"Dimensions: {arr.ndim}")  # 1
print(f"Size: {arr.size}")        # 4
```

### Essential Array Properties
- **shape**: Dimensions of the array → *Critical for debugging*
- **dtype**: Data type → *Controls memory usage and precision*
- **ndim**: Number of dimensions → *Helps visualize your data structure*
- **size**: Total number of elements → *Useful for memory calculations*

### Data Types That Matter in AI/ML

```python
# Integer types
int32_arr = np.array([1, 2, 3], dtype=np.int32)    # 32-bit integers
int64_arr = np.array([1, 2, 3], dtype=np.int64)    # 64-bit integers (default)

# Float types (MOST IMPORTANT for AI/ML)
float32_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # Single precision (GPU friendly)
float64_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # Double precision (default)

# Boolean (for masking)
bool_arr = np.array([True, False, True])
```

**AI/ML Insight**: Use `float32` for deep learning (GPU optimization) and `float64` for scientific computing (higher precision).

### Practice Exercise 2: Array Basics
```python
# Create these arrays and explore their properties:
# 1. A 1D array of 10 random floats between 0 and 1
# 2. A 2D array (3×4) of integers from 1 to 12
# 3. A 3D array (2×3×4) of zeros

# Your solutions:
arr1 = np.random.random(10)
arr2 = np.arange(1, 13).reshape(3, 4)
arr3 = np.zeros((2, 3, 4))

# Exploration questions:
# - What happens when you change dtype to int32?
# - How much memory does each array use? (hint: arr.nbytes)
```

---

## 3. Array Creation & Manipulation: Building Your Data {#array-creation}

### Creation Functions You'll Use Daily

#### From Lists and Sequences
```python
# Basic creation
list_to_array = np.array([1, 2, 3, 4, 5])
nested_list = np.array([[1, 2], [3, 4], [5, 6]])

# Range functions
linear_space = np.linspace(0, 10, 100)    # 100 points from 0 to 10
integer_range = np.arange(0, 20, 2)       # 0, 2, 4, ..., 18
```

#### Initialization Functions (The Power Trio)
```python
# Zeros - for initializing weights, placeholders
zeros_2d = np.zeros((3, 4))               # 3×4 matrix of zeros
zeros_like = np.zeros_like(existing_arr)  # Same shape as existing_arr

# Ones - for creating masks, initial biases
ones_3d = np.ones((2, 3, 4))
identity = np.eye(5)                      # 5×5 identity matrix

# Random arrays - THE foundation of ML
random_uniform = np.random.random((100, 784))     # Uniform [0,1)
random_normal = np.random.randn(64, 128)          # Standard normal
random_range = np.random.randint(0, 10, (5, 5))  # Random integers
```

**AI/ML Connection**: Random initialization prevents symmetric weight updates in neural networks.

#### Advanced Creation for AI/ML
```python
# Create training data batches
batch_size, features = 32, 128
X_batch = np.random.randn(batch_size, features)

# Create one-hot encoded labels
num_classes = 10
labels = np.random.randint(0, num_classes, batch_size)
y_onehot = np.eye(num_classes)[labels]  # Clever one-hot encoding trick!
```

### Reshaping: Your Shape-Shifting Power

```python
# Original array
arr = np.arange(24)  # [0, 1, 2, ..., 23]

# Reshape transformations
matrix_2d = arr.reshape(4, 6)      # 4 rows, 6 columns
matrix_3d = arr.reshape(2, 3, 4)   # 2 matrices of 3×4
vector = arr.reshape(-1)           # Flatten to 1D (-1 means "figure it out")
column_vector = arr.reshape(-1, 1) # Column vector (24, 1)

# Key insight: Total elements must remain the same!
```

**Critical AI/ML Application**: Reshaping is essential for:
- Flattening images: (28, 28) → (784,) for dense layers
- Preparing RNN inputs: (batch, time, features)
- Converting model outputs: (batch, classes) → predictions

### Practice Exercise 3: Data Preparation
```python
# Simulate real AI/ML data preparation tasks:

# 1. Create a dataset of 1000 samples with 20 features
# 2. Reshape it to simulate image data (assume square images)
# 3. Create corresponding random labels (0-9)
# 4. Create a validation split (20% of data)

# Challenge solution:
n_samples, n_features = 1000, 20
X = np.random.randn(n_samples, n_features)

# If n_features allows square images
img_size = int(np.sqrt(n_features))  # 4 (since sqrt(20) ≈ 4.47, but let's use 400 features)
X_corrected = np.random.randn(1000, 400)
X_images = X_corrected.reshape(1000, 20, 20)

# Labels
y = np.random.randint(0, 10, 1000)

# Train-validation split
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]
```

---

## 4. Indexing & Slicing: Accessing Your Universe {#indexing-slicing}

### Basic Indexing: Your Navigation System

```python
# 1D indexing
arr = np.array([10, 20, 30, 40, 50])
first = arr[0]          # 10
last = arr[-1]          # 50
middle_three = arr[1:4] # [20, 30, 40]

# 2D indexing (row, column)
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
element = matrix[1, 2]     # 6 (row 1, column 2)
first_row = matrix[0, :]   # [1, 2, 3]
last_column = matrix[:, -1] # [3, 6, 9]
```

### Advanced Indexing: The AI/ML Game Changers

#### Boolean Indexing (Data Filtering)
```python
data = np.array([1, -2, 3, -4, 5, -6])

# Filter positive values
positive_mask = data > 0        # [True, False, True, False, True, False]
positive_values = data[positive_mask]  # [1, 3, 5]

# One-liner
positive_direct = data[data > 0]  # [1, 3, 5]
```

**AI/ML Power Move**: Filter outliers, select valid samples, create training masks
```python
# Remove outliers beyond 2 standard deviations
clean_data = data[np.abs(data - np.mean(data)) < 2 * np.std(data)]

# Select samples based on labels
X_class_0 = X[y == 0]  # All samples with label 0
```

#### Fancy Indexing (Array Indexing)
```python
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 2, 4])
selected = arr[indices]  # [10, 30, 50]

# 2D fancy indexing
matrix = np.random.randn(5, 3)
row_indices = [0, 2, 4]
col_indices = [1, 2, 0]
values = matrix[row_indices, col_indices]  # Specific (row, col) pairs
```

**AI/ML Application**: Batch sampling, data augmentation, model evaluation
```python
# Random batch sampling
batch_indices = np.random.choice(len(X), size=32, replace=False)
X_batch = X[batch_indices]
y_batch = y[batch_indices]
```

### Practice Exercise 4: Data Manipulation
```python
# Scenario: You have a dataset with features and need to preprocess it
np.random.seed(42)  # For reproducibility
data = np.random.randn(100, 10)  # 100 samples, 10 features
labels = np.random.randint(0, 3, 100)  # 3 classes

# Tasks:
# 1. Extract all samples belonging to class 1
# 2. Get the first 5 features of these samples
# 3. Remove samples where any feature exceeds 2 standard deviations
# 4. Create a random sample of 20 samples from the cleaned data

# Solutions:
class_1_samples = data[labels == 1]
first_5_features = class_1_samples[:, :5]

# Outlier removal
outlier_mask = np.any(np.abs(data) > 2 * np.std(data, axis=0), axis=1)
clean_data = data[~outlier_mask]

# Random sampling
random_indices = np.random.choice(len(clean_data), size=20, replace=False)
random_sample = clean_data[random_indices]
```

---

## 5. Mathematical Operations: The Heart of AI/ML {#mathematical-operations}

### Element-wise Operations: Your Daily Bread

```python
# Basic arithmetic (vectorized - super fast!)
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

addition = a + b        # [6, 8, 10, 12]
multiplication = a * b  # [5, 12, 21, 32]
power = a ** 2         # [1, 4, 9, 16]
sqrt = np.sqrt(a)      # [1.0, 1.414, 1.732, 2.0]
```

**Performance Insight**: This is 10-100x faster than Python loops!

### Universal Functions (ufuncs): Mathematical Superpowers

```python
# Trigonometric functions
angles = np.linspace(0, 2*np.pi, 100)
sine_wave = np.sin(angles)
cosine_wave = np.cos(angles)

# Exponential and logarithmic (crucial for ML)
x = np.array([1, 2, 3, 4, 5])
exponential = np.exp(x)      # e^x
natural_log = np.log(x)      # ln(x)
log_base_10 = np.log10(x)    # log10(x)

# Activation functions you'll implement
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)
```

### Aggregation Operations: Understanding Your Data

```python
data = np.random.randn(100, 50)  # 100 samples, 50 features

# Along different axes
feature_means = np.mean(data, axis=0)    # Mean of each feature (shape: 50,)
sample_means = np.mean(data, axis=1)     # Mean of each sample (shape: 100,)
overall_mean = np.mean(data)             # Single number

# Essential statistics
std_devs = np.std(data, axis=0)          # Standard deviation per feature
mins = np.min(data, axis=0)              # Minimum per feature
maxs = np.max(data, axis=0)              # Maximum per feature
```

**AI/ML Critical Insight**: Axis understanding is crucial:
- `axis=0`: Operations along rows (column-wise results)
- `axis=1`: Operations along columns (row-wise results)
- `axis=None`: Operations on entire array (single result)

### Practice Exercise 5: Feature Scaling
```python
# Implement standard scaling (z-score normalization)
# This is fundamental preprocessing for many ML algorithms

np.random.seed(42)
raw_features = np.random.randn(1000, 20) * 100 + 50  # Random data with mean=50, std=100

# Task: Implement standardization manually
# Formula: (x - mean) / std

def standardize(X):
    """
    Standardize features to have mean=0 and std=1
    """
    # Your implementation here
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

# Test your implementation
X_scaled = standardize(raw_features)
print(f"Original mean: {np.mean(raw_features, axis=0)[:3]}")  # Should be ~50
print(f"Scaled mean: {np.mean(X_scaled, axis=0)[:3]}")       # Should be ~0
print(f"Scaled std: {np.std(X_scaled, axis=0)[:3]}")        # Should be ~1
```

---

## 6. Broadcasting: NumPy's Secret Weapon {#broadcasting}

### What Is Broadcasting?
Broadcasting allows NumPy to perform operations on arrays of different shapes without explicitly reshaping them. This is **pure magic** for AI/ML workflows.

### Broadcasting Rules (Learn These!)
1. Arrays are aligned from the **rightmost dimension**
2. Dimensions of size 1 are **stretched** to match
3. Missing dimensions are treated as size 1

### Visual Examples

```python
# Example 1: Adding a bias to all samples
X = np.random.randn(100, 3)  # 100 samples, 3 features
bias = np.array([0.1, 0.2, 0.3])  # Shape: (3,)

# Broadcasting magic!
X_biased = X + bias  # bias broadcasts to (100, 3)
```

```
Original shapes:
X:    (100, 3)
bias: (3,)

After broadcasting:
X:    (100, 3)
bias: (100, 3)  ← Automatically expanded!
```

### Advanced Broadcasting Patterns

```python
# Pattern 1: Feature scaling across samples
X = np.random.randn(1000, 784)  # 1000 images, 784 pixels each
pixel_means = np.mean(X, axis=0)  # Shape: (784,)
X_centered = X - pixel_means      # Broadcasting: (1000, 784) - (784,)

# Pattern 2: Sample normalization
sample_norms = np.linalg.norm(X, axis=1, keepdims=True)  # Shape: (1000, 1)
X_normalized = X / sample_norms   # Broadcasting: (1000, 784) / (1000, 1)

# Pattern 3: Distance calculations
point_a = np.array([1, 2, 3])     # Shape: (3,)
points_b = np.random.randn(100, 3)  # Shape: (100, 3)
distances = np.linalg.norm(points_b - point_a, axis=1)  # Broadcasting magic!
```

### Common Broadcasting Errors and Fixes

```python
# ERROR: Incompatible shapes
try:
    result = np.array([[1, 2, 3]]) + np.array([[1], [2], [3], [4]])  # (1,3) + (4,1) → (4,3)
except ValueError as e:
    print("Error:", e)

# SOLUTION: Use reshape or newaxis
a = np.array([1, 2, 3])          # Shape: (3,)
b = np.array([1, 2, 3, 4])       # Shape: (4,)
result = a + b[:, np.newaxis]    # Now compatible: (3,) + (4,1) → (4,3)
```

### Practice Exercise 6: Broadcasting Mastery
```python
# Scenario: Implement batch normalization (core of modern deep learning)
# Given: batch of data (N, D) where N=batch_size, D=features
# Goal: Normalize each feature across the batch

def batch_normalize(X, epsilon=1e-8):
    """
    Implement batch normalization
    X: shape (N, D)
    Returns: normalized X with mean=0, std=1 per feature
    """
    # Your implementation using broadcasting
    batch_mean = np.mean(X, axis=0, keepdims=True)  # Shape: (1, D)
    batch_var = np.var(X, axis=0, keepdims=True)    # Shape: (1, D)
    X_normalized = (X - batch_mean) / np.sqrt(batch_var + epsilon)
    return X_normalized

# Test with simulated neural network layer output
layer_output = np.random.randn(64, 128)  # 64 samples, 128 neurons
normalized_output = batch_normalize(layer_output)

print(f"Original mean per feature: {np.mean(layer_output, axis=0)[:5]}")
print(f"Normalized mean per feature: {np.mean(normalized_output, axis=0)[:5]}")
```

---

## 7. Linear Algebra: The AI/ML Powerhouse {#linear-algebra}

### Matrix Multiplication: The Engine of AI

```python
# The @ operator (Python 3.5+) - your new best friend
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = A @ B  # Shape: (3, 5)

# Equivalent ways
C_dot = np.dot(A, B)
C_matmul = np.matmul(A, B)
```

### Neural Network Forward Pass Implementation

```python
def neural_network_forward(X, W1, b1, W2, b2):
    """
    Simple 2-layer neural network forward pass
    X: input data (batch_size, input_features)
    W1, W2: weight matrices
    b1, b2: bias vectors
    """
    # Layer 1
    z1 = X @ W1 + b1        # Linear transformation
    a1 = np.maximum(0, z1)  # ReLU activation
    
    # Layer 2 (output)
    z2 = a1 @ W2 + b2       # Linear transformation
    a2 = sigmoid(z2)        # Sigmoid activation
    
    return a2

# Example usage
batch_size, input_dim, hidden_dim, output_dim = 32, 784, 128, 10

X = np.random.randn(batch_size, input_dim)
W1 = np.random.randn(input_dim, hidden_dim) * 0.01  # Small random weights
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros(output_dim)

predictions = neural_network_forward(X, W1, b1, W2, b2)
print(f"Predictions shape: {predictions.shape}")  # (32, 10)
```

### Essential Linear Algebra Operations

```python
# Matrix properties
A = np.random.randn(5, 5)
A_transpose = A.T                    # Transpose
A_inverse = np.linalg.inv(A)         # Inverse (if exists)
determinant = np.linalg.det(A)       # Determinant
rank = np.linalg.matrix_rank(A)      # Rank

# Eigenvalues and eigenvectors (PCA foundation)
eigenvalues, eigenvectors = np.linalg.eig(A)

# Singular Value Decomposition (SVD) - dimensionality reduction
U, s, Vt = np.linalg.svd(A)
```

### Vector Operations for Similarity and Distance

```python
# Dot product (similarity measure)
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
similarity = np.dot(v1, v2)  # 32

# Cosine similarity (normalized dot product)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Batch cosine similarity (vectorized)
def batch_cosine_similarity(X, y):
    """
    Compute cosine similarity between each row of X and vector y
    """
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    y_norm = y / np.linalg.norm(y)
    return X_norm @ y_norm
```

### Practice Exercise 7: Implement PCA from Scratch
```python
def pca_from_scratch(X, n_components):
    """
    Implement Principal Component Analysis
    X: data matrix (n_samples, n_features)
    n_components: number of principal components to keep
    """
    # Step 1: Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Step 4: Sort by eigenvalues (descending)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]
    
    # Step 5: Transform data
    X_transformed = X_centered @ top_eigenvectors
    
    return X_transformed, top_eigenvectors

# Test your PCA implementation
test_data = np.random.randn(200, 10)
X_pca, components = pca_from_scratch(test_data, n_components=3)
print(f"Original shape: {test_data.shape}")
print(f"PCA shape: {X_pca.shape}")
print(f"Variance explained: {np.var(X_pca, axis=0)}")
```

---

## 8. Statistical Operations: Understanding Your Data {#statistical-operations}

### Descriptive Statistics: Your Data Detective Tools

```python
# Generate sample dataset
np.random.seed(42)
dataset = np.random.exponential(scale=2.0, size=(1000, 5))  # Right-skewed data

# Central tendency
means = np.mean(dataset, axis=0)      # [2.01, 1.98, 2.05, 1.99, 2.02]
medians = np.median(dataset, axis=0)  # More robust to outliers
modes = scipy.stats.mode(dataset, axis=0)  # Requires scipy

# Spread measures
stds = np.std(dataset, axis=0)        # Standard deviation
vars = np.var(dataset, axis=0)        # Variance
ranges = np.ptp(dataset, axis=0)      # Peak-to-peak (max - min)

# Percentiles (quartiles)
q25 = np.percentile(dataset, 25, axis=0)  # 1st quartile
q75 = np.percentile(dataset, 75, axis=0)  # 3rd quartile
iqr = q75 - q25                           # Interquartile range
```

### Correlation Analysis: Finding Relationships

```python
# Correlation matrix
correlation_matrix = np.corrcoef(dataset.T)  # Features as variables

# Covariance matrix
covariance_matrix = np.cov(dataset.T)

# Custom correlation function
def correlation(x, y):
    """Compute Pearson correlation coefficient"""
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    
    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    
    return numerator / denominator
```

### Probability and Sampling: The Foundation of ML

```python
# Random sampling strategies
def stratified_sample(X, y, n_samples_per_class):
    """
    Create stratified sample maintaining class balance
    """
    unique_classes = np.unique(y)
    stratified_X = []
    stratified_y = []
    
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        sampled_indices = np.random.choice(
            class_indices, 
            size=min(n_samples_per_class, len(class_indices)), 
            replace=False
        )
        stratified_X.append(X[sampled_indices])
        stratified_y.append(y[sampled_indices])
    
    return np.vstack(stratified_X), np.hstack(stratified_y)

# Bootstrap sampling for confidence intervals
def bootstrap_mean(data, n_bootstrap=1000):
    """
    Compute bootstrap confidence interval for mean
    """
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    return np.array(bootstrap_means)
```

### Practice Exercise 8: Data Quality Analysis
```python
# Scenario: You're given a messy real-world dataset
np.random.seed(42)

# Simulate realistic messy data
n_samples, n_features = 1000, 20
clean_data = np.random.randn(n_samples, n_features)

# Add some realistic problems
messy_data = clean_data.copy()
# Add missing values (represented as NaN)
missing_indices = np.random.choice(n_samples * n_features, size=200, replace=False)
flat_view = messy_data.flatten()
flat_view[missing_indices] = np.nan
messy_data = flat_view.reshape(n_samples, n_features)

# Add outliers
outlier_indices = np.random.choice(n_samples, size=50, replace=False)
messy_data[outlier_indices] += np.random.randn(50, n_features) * 10

# Your tasks:
# 1. Identify features with missing values
# 2. Compute percentage of missing values per feature
# 3. Identify outliers using IQR method
# 4. Create a cleaned version of the dataset

# Solutions:
# 1. Features with missing values
has_missing = np.isnan(messy_data).any(axis=0)
print(f"Features with missing values: {np.where(has_missing)[0]}")

# 2. Missing percentage per feature
missing_percentage = np.isnan(messy_data).mean(axis=0) * 100
print(f"Missing percentages: {missing_percentage[:5]}")

# 3. Outlier detection using IQR
def detect_outliers_iqr(X):
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Find outliers
    outliers = (X < lower_bound) | (X > upper_bound)
    return outliers

outlier_mask = detect_outliers_iqr(messy_data)

# 4. Create cleaned dataset
def clean_dataset(X):
    # Remove samples with too many missing values (>50%)
    sample_missing_rate = np.isnan(X).mean(axis=1)
    valid_samples = sample_missing_rate < 0.5
    X_clean = X[valid_samples]
    
    # Impute remaining missing values with feature means
    feature_means = np.nanmean(X_clean, axis=0)
    for i in range(X_clean.shape[1]):
        feature_missing = np.isnan(X_clean[:, i])
        X_clean[feature_missing, i] = feature_means[i]
    
    return X_clean

clean_dataset_result = clean_dataset(messy_data)
print(f"Original shape: {messy_data.shape}")
print(f"Cleaned shape: {clean_dataset_result.shape}")
```

---