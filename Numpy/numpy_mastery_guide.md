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
9. [Advanced Techniques: Performance & Optimization](#advanced-techniques)
10. [Real-World AI/ML Projects](#real-world-projects)

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

## 9. Advanced Techniques: Performance & Optimization {#advanced-techniques}

### Memory Layout and Performance

```python
# Row-major (C-style) vs Column-major (Fortran-style)
large_array = np.random.randn(1000, 1000)

# Check memory layout
print(f"C-contiguous: {large_array.flags['C_CONTIGUOUS']}")
print(f"F-contiguous: {large_array.flags['F_CONTIGUOUS']}")

# Performance tip: Access patterns matter!
# Fast: row-wise access for C-contiguous arrays
row_sum = np.sum(large_array, axis=1)  # Sum across columns (fast)

# Slower: column-wise access for C-contiguous arrays  
col_sum = np.sum(large_array, axis=0)  # Sum across rows (slower)
```

### Vectorization: Eliminating Loops

```python
# SLOW: Python loop version
def slow_euclidean_distances(X, y):
    distances = []
    for i in range(len(X)):
        dist = 0
        for j in range(len(X[i])):
            dist += (X[i][j] - y[j]) ** 2
        distances.append(np.sqrt(dist))
    return np.array(distances)

# FAST: Vectorized version
def fast_euclidean_distances(X, y):
    return np.sqrt(np.sum((X - y) ** 2, axis=1))

# Performance comparison
X_test = np.random.randn(10000, 100)
y_test = np.random.randn(100)

# The vectorized version is 50-100x faster!
```

### Advanced Array Manipulation

```python
# Stacking and concatenation
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Horizontal stacking (side by side)
h_stack = np.hstack([a, b])     # Shape: (2, 4)
h_concat = np.concatenate([a, b], axis=1)  # Same result

# Vertical stacking (top to bottom)
v_stack = np.vstack([a, b])     # Shape: (4, 2)
v_concat = np.concatenate([a, b], axis=0)  # Same result

# Split arrays
split_arrays = np.split(v_stack, 2, axis=0)  # Split into 2 equal parts
```

### Memory-Efficient Operations

```python
# In-place operations (save memory)
large_array = np.random.randn(10000, 1000)

# Memory inefficient (creates new array)
result = large_array * 2

# Memory efficient (modifies existing array)
large_array *= 2  # In-place multiplication

# Views vs Copies
original = np.arange(20).reshape(4, 5)
view = original[1:3, 2:4]      # Creates a view (shares memory)
copy = original[1:3, 2:4].copy()  # Creates a copy (new memory)

# Modify view affects original
view[0, 0] = 999
print(original[1, 2])  # 999! View affected original

# Copy is independent
copy[0, 0] = 777
print(original)  # Original unchanged
```

### Practice Exercise 9: Optimized K-Means Implementation
```python
def kmeans_numpy(X, k, max_iters=100, tol=1e-4):
    """
    Implement K-means clustering using optimized NumPy operations
    """
    n_samples, n_features = X.shape
    
    # Initialize centroids randomly
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for iteration in range(max_iters):
        # Compute distances to all centroids (vectorized)
        # Shape: (n_samples, k)
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # Assign clusters
        cluster_assignments = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([
            X[cluster_assignments == i].mean(axis=0) if np.any(cluster_assignments == i) 
            else centroids[i] 
            for i in range(k)
        ])
        
        # Check convergence
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        if centroid_shift < tol:
            break
            
        centroids = new_centroids
    
    return cluster_assignments, centroids

# Test implementation
test_data = np.random.randn(1000, 2)
clusters, centers = kmeans_numpy(test_data, k=3)
print(f"Converged after optimization")
print(f"Final centroids:\n{centers}")
```

---

## 10. Broadcasting Deep Dive: Advanced Patterns {#broadcasting}

### Multi-dimensional Broadcasting Patterns

```python
# Pattern 1: Attention mechanism (simplified)
def simple_attention(Q, K, V):
    """
    Simplified attention mechanism
    Q: queries (seq_len, d_model)
    K: keys (seq_len, d_model)  
    V: values (seq_len, d_model)
    """
    # Compute attention scores
    scores = Q @ K.T  # (seq_len, seq_len)
    
    # Apply softmax
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    
    # Apply attention to values
    output = attention_weights @ V
    return output

# Pattern 2: Batch matrix operations
batch_size, seq_len, d_model = 32, 128, 512
Q_batch = np.random.randn(batch_size, seq_len, d_model)
K_batch = np.random.randn(batch_size, seq_len, d_model)
V_batch = np.random.randn(batch_size, seq_len, d_model)

# Batch attention (using np.matmul for batch operations)
scores_batch = np.matmul(Q_batch, K_batch.transpose(0, 2, 1))
attention_batch = np.matmul(scores_batch, V_batch)
```

### Conditional Operations and Masking

```python
# Where function: conditional selection
condition = np.array([True, False, True, False])
x = np.array([1, 2, 3, 4])
y = np.array([10, 20, 30, 40])
result = np.where(condition, x, y)  # [1, 20, 3, 40]

# Clipping (essential for gradient clipping)
gradients = np.random.randn(100) * 10
clipped_gradients = np.clip(gradients, -5, 5)  # Clip to [-5, 5]

# Advanced masking for data cleaning
def remove_extreme_values(X, threshold_std=3):
    """Remove samples with extreme values"""
    z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
    extreme_mask = np.any(z_scores > threshold_std, axis=1)
    return X[~extreme_mask]
```

---

## 11. Real-World AI/ML Projects {#real-world-projects}

### Project 1: Image Preprocessing Pipeline

```python
def image_preprocessing_pipeline(images):
    """
    Complete image preprocessing for ML
    images: (N, H, W, C) or (N, H, W) array
    """
    # Normalize pixel values to [0, 1]
    images_normalized = images.astype(np.float32) / 255.0
    
    # Center around mean
    images_centered = images_normalized - np.mean(images_normalized)
    
    # Add noise for data augmentation
    noise = np.random.normal(0, 0.01, images_centered.shape)
    images_augmented = images_centered + noise
    
    # Clip to valid range
    images_final = np.clip(images_augmented, 0, 1)
    
    return images_final

# Simulate image data
fake_images = np.random.randint(0, 256, (1000, 28, 28), dtype=np.uint8)
processed_images = image_preprocessing_pipeline(fake_images)
```

### Project 2: Feature Engineering Toolkit

```python
class FeatureEngineer:
    """Advanced feature engineering using pure NumPy"""
    
    def __init__(self):
        self.scalers = {}
    
    def polynomial_features(self, X, degree=2):
        """Generate polynomial features"""
        n_samples, n_features = X.shape
        features = [X]
        
        for d in range(2, degree + 1):
            # Generate all combinations for degree d
            poly_features = X ** d
            features.append(poly_features)
        
        return np.hstack(features)
    
    def interaction_features(self, X):
        """Create interaction features (pairwise products)"""
        n_samples, n_features = X.shape
        interactions = []
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction = X[:, i] * X[:, j]
                interactions.append(interaction.reshape(-1, 1))
        
        return np.hstack([X] + interactions) if interactions else X
    
    def fit_transform_standard_scaler(self, X):
        """Fit and transform standard scaler"""
        self.scalers['mean'] = np.mean(X, axis=0)
        self.scalers['std'] = np.std(X, axis=0)
        return (X - self.scalers['mean']) / self.scalers['std']
    
    def transform_standard_scaler(self, X):
        """Transform using fitted scaler"""
        return (X - self.scalers['mean']) / self.scalers['std']

# Usage example
fe = FeatureEngineer()
X_train = np.random.randn(800, 10)
X_test = np.random.randn(200, 10)

# Feature engineering pipeline
X_train_poly = fe.polynomial_features(X_train, degree=2)
X_train_scaled = fe.fit_transform_standard_scaler(X_train_poly)

X_test_poly = fe.polynomial_features(X_test, degree=2)
X_test_scaled = fe.transform_standard_scaler(X_test_poly)
```

### Project 3: Linear Regression from Scratch

```python
class LinearRegressionNumPy:
    """Complete linear regression implementation using NumPy"""
    
    def __init__(self, learning_rate=0.01, max_iters=1000):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iters):
            # Forward pass
            y_pred = X @ self.weights + self.bias
            
            # Compute cost (MSE)
            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (2 / n_samples) * X.T @ (y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Early stopping
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < 1e-8:
                break
    
    def predict(self, X):
        """Make predictions"""
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Test the implementation
np.random.seed(42)
X_train = np.random.randn(1000, 5)
true_weights = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
y_train = X_train @ true_weights + np.random.randn(1000) * 0.1

model = LinearRegressionNumPy(learning_rate=0.1)
model.fit(X_train, y_train)

print(f"True weights: {true_weights}")
print(f"Learned weights: {model.weights}")
print(f"R² score: {model.score(X_train, y_train):.4f}")
```

---

## 12. Performance Optimization: Speed That Matters {#performance-optimization}

### Profiling and Optimization Techniques

```python
import time

def profile_function(func, *args, **kwargs):
    """Simple profiling utility"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"{func.__name__} took {end_time - start_time:.6f} seconds")
    return result

# Optimization technique 1: Use built-in functions
def slow_sum_of_squares(arr):
    total = 0
    for element in arr.flatten():
        total += element ** 2
    return total

def fast_sum_of_squares(arr):
    return np.sum(arr ** 2)

# Test performance difference
test_array = np.random.randn(10000, 100)
# profile_function(slow_sum_of_squares, test_array)  # ~2 seconds
# profile_function(fast_sum_of_squares, test_array)  # ~0.002 seconds
```

### Memory-Efficient Computations

```python
# Technique 1: Process data in chunks
def process_large_dataset_chunked(data, chunk_size=1000):
    """Process large datasets without loading everything into memory"""
    n_samples = data.shape[0]
    results = []
    
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        chunk = data[start_idx:end_idx]
        
        # Process chunk (example: normalize)
        chunk_normalized = (chunk - np.mean(chunk, axis=0)) / np.std(chunk, axis=0)
        results.append(chunk_normalized)
    
    return np.vstack(results)

# Technique 2: Use appropriate data types
def optimize_memory_usage(X):
    """Reduce memory usage by choosing appropriate dtypes"""
    if np.all(X == X.astype(np.int8)):
        return X.astype(np.int8)  # -128 to 127
    elif np.all(X == X.astype(np.int16)):
        return X.astype(np.int16)  # -32768 to 32767
    elif X.dtype == np.float64:
        return X.astype(np.float32)  # Often sufficient precision
    return X
```

---

## 13. Common NumPy Patterns for AI/ML {#common-patterns}

### Pattern 1: Data Shuffling and Batching

```python
def create_batches(X, y, batch_size, shuffle=True):
    """Create training batches"""
    n_samples = len(X)
    
    if shuffle:
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
    else:
        X_shuffled, y_shuffled = X, y
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        yield X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]

# Usage in training loop
X_train = np.random.randn(10000, 50)
y_train = np.random.randint(0, 5, 10000)

for epoch in range(10):
    for X_batch, y_batch in create_batches(X_train, y_train, batch_size=64):
        # Training step here
        pass
```

### Pattern 2: One-Hot Encoding and Label Processing

```python
def one_hot_encode(labels, num_classes=None):
    """Convert integer labels to one-hot encoding"""
    if num_classes is None:
        num_classes = np.max(labels) + 1
    
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

# Alternative elegant method
def one_hot_elegant(labels, num_classes=None):
    if num_classes is None:
        num_classes = np.max(labels) + 1
    return np.eye(num_classes)[labels]

# Usage
labels = np.array([0, 1, 2, 1, 0])
one_hot = one_hot_elegant(labels, num_classes=3)
print(one_hot)
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [0. 1. 0.]
#  [1. 0. 0.]]
```

### Pattern 3: Confusion Matrix and Metrics

```python
def confusion_matrix_numpy(y_true, y_pred, num_classes):
    """Compute confusion matrix using NumPy"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    
    return cm

def classification_metrics(y_true, y_pred, num_classes):
    """Compute precision, recall, F1-score"""
    cm = confusion_matrix_numpy(y_true, y_pred, num_classes)
    
    # Per-class metrics
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Handle division by zero
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1_score = np.nan_to_num(f1_score)
    
    return {
        'precision': precision,
        'recall': recall, 
        'f1_score': f1_score,
        'confusion_matrix': cm
    }
```

---

## 14. Advanced Linear Algebra for Deep Learning {#advanced-linear-algebra}

### Matrix Factorization Techniques

```python
# Singular Value Decomposition (SVD) for dimensionality reduction
def truncated_svd(X, n_components):
    """Implement truncated SVD for dimensionality reduction"""
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute SVD
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Keep only top n_components
    U_truncated = U[:, :n_components]
    s_truncated = s[:n_components]
    Vt_truncated = Vt[:n_components, :]
    
    # Transform data
    X_transformed = U_truncated * s_truncated
    
    return X_transformed, (U_truncated, s_truncated, Vt_truncated)

# QR decomposition for numerical stability
def qr_solver(A, b):
    """Solve Ax = b using QR decomposition (more stable than normal equations)"""
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R, Q.T @ b)
```

### Eigenvalue Problems in ML

```python
def power_iteration(A, num_iterations=100):
    """Find largest eigenvalue and eigenvector using power iteration"""
    n = A.shape[0]
    
    # Random initial vector
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(num_iterations):
        # Power iteration step
        Av = A @ v
        v = Av / np.linalg.norm(Av)
        
        # Eigenvalue estimate
        eigenvalue = v.T @ A @ v
    
    return eigenvalue, v

# Application: Principal Component Analysis (PCA) implementation
def pca_eigenvalue_method(X, n_components):
    """PCA using eigenvalue decomposition"""
    # Center data
    X_centered = X - np.mean(X, axis=0)
    
    # Covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top components
    principal_components = eigenvectors[:, :n_components]
    
    # Transform data
    X_pca = X_centered @ principal_components
    
    # Explained variance ratio
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return X_pca, explained_variance_ratio
```

---

## 15. Optimization and Numerical Stability {#optimization}

### Numerical Stability Techniques

```python
# Stable softmax implementation (prevents overflow)
def stable_softmax(x):
    """Numerically stable softmax"""
    # Subtract max for numerical stability
    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Log-sum-exp trick for log probabilities
def log_sum_exp(x):
    """Numerically stable log(sum(exp(x)))"""
    max_x = np.max(x, axis=-1, keepdims=True)
    return max_x + np.log(np.sum(np.exp(x - max_x), axis=-1, keepdims=True))

# Stable gradient computation
def stable_gradient_norm(gradients, epsilon=1e-8):
    """Compute gradient norm with numerical stability"""
    return np.sqrt(np.sum(gradients ** 2) + epsilon)
```

### Advanced Optimization Techniques

```python
class AdamOptimizer:
    """Adam optimizer implementation in NumPy"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def update(self, params, gradients):
        """Update parameters using Adam optimizer"""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        
        # Compute bias-corrected moment estimates
        m_corrected = self.m / (1 - self.beta1 ** self.t)
        v_corrected = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        params -= self.lr * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
        
        return params
```

---

## 16. Final Project: Neural Network from Scratch {#final-project}

### Complete Implementation

```python
class NeuralNetworkNumPy:
    """Multi-layer neural network using only NumPy"""
    
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Xavier initialization
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current_input @ w + b
            self.z_values.append(z)
            
            # Apply activation (ReLU for hidden layers, sigmoid for output)
            if i < len(self.weights) - 1:
                a = np.maximum(0, z)  # ReLU
            else:
                a = stable_softmax(z)  # Softmax for multiclass
            
            self.activations.append(a)
            current_input = a
        
        return self.activations[-1]
    
    def backward(self, X, y):
        """Backward propagation"""
        m = X.shape[0]  # batch size
        
        # Convert labels to one-hot if needed
        if len(y.shape) == 1:
            y_onehot = np.eye(self.layer_sizes[-1])[y]
        else:
            y_onehot = y
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        delta = self.activations[-1] - y_onehot
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradients for current layer
            dW[i] = self.activations[i].T @ delta / m
            db[i] = np.mean(delta, axis=0)
            
            # Propagate error to previous layer
            if i > 0:
                delta = (delta @ self.weights[i].T) * (self.z_values[i-1] > 0)  # ReLU derivative
        
        return dW, db
    
    def update_parameters(self, dW, db):
        """Update weights and biases"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def train(self, X, y, epochs=100, batch_size=32):
        """Training loop"""
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Process in batches
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions = self.forward(X_batch)
                
                # Compute loss (cross-entropy)
                if len(y_batch.shape) == 1:
                    y_onehot = np.eye(self.layer_sizes[-1])[y_batch]
                else:
                    y_onehot = y_batch
                
                batch_loss = -np.mean(np.sum(y_onehot * np.log(predictions + 1e-15), axis=1))
                epoch_loss += batch_loss
                n_batches += 1
                
                # Backward pass
                dW, db = self.backward(X_batch, y_batch)
                
                # Update parameters
                self.update_parameters(dW, db)
            
            if epoch % 10 == 0:
                avg_loss = epoch_loss / n_batches
                accuracy = self.accuracy(X, y)
                print(f"
    