# Compressive Hyperspectral Imaging

A Python implementation of compressive sensing techniques for hyperspectral imaging, including coded aperture snapshot spectral imaging (CASSI) with multiple reconstruction algorithms.

## Overview

Compressive Hyperspectral Imaging (CHI) is a technique that captures hyperspectral images using fewer measurements than traditional methods by exploiting the inherent sparsity of hyperspectral data. This implementation provides:

- **Measurement Matrix Generation**: Random Gaussian matrices for compressive sensing
- **Compression**: Efficient compression of hyperspectral cubes
- **Reconstruction Algorithms**:
  - L1 Minimization (LASSO)
  - Orthogonal Matching Pursuit (OMP)
  - Total Variation (TV) minimization

## Features

- 🎯 Multiple reconstruction algorithms for flexibility
- 📊 Quality metrics (PSNR, RMSE, Relative Error)
- 🔬 Synthetic data generation for testing
- ⚡ Efficient numpy-based implementation
- 📖 Comprehensive documentation and examples

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Karen-Chou/1.git
cd 1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the example demonstration:

```bash
python example.py
```

This will demonstrate the compression and reconstruction of a synthetic hyperspectral image using different algorithms.

## Usage

### Basic Usage

```python
import numpy as np
from compressive_hsi import CompressiveHSI, generate_synthetic_hsi, compute_metrics

# Generate synthetic hyperspectral image
hsi_cube = generate_synthetic_hsi(height=32, width=32, spectral_bands=31)

# Initialize compressive sensing system
chi_system = CompressiveHSI(
    spatial_dims=(32, 32),
    spectral_bands=31,
    compression_ratio=0.3
)

# Compress
measurements = chi_system.compress(hsi_cube)

# Reconstruct using L1 minimization
reconstructed = chi_system.reconstruct_l1(measurements)

# Compute quality metrics
metrics = compute_metrics(hsi_cube, reconstructed)
print(f"PSNR: {metrics['PSNR']:.2f} dB")
```

### Advanced Usage

#### Using Different Reconstruction Algorithms

```python
# L1 Minimization (LASSO)
reconstructed_l1 = chi_system.reconstruct_l1(measurements, max_iter=1000)

# Orthogonal Matching Pursuit
reconstructed_omp = chi_system.reconstruct_omp(measurements, sparsity=100)

# Total Variation
reconstructed_tv = chi_system.reconstruct_tval3(measurements, max_iter=100)
```

#### Custom Measurement Matrix

The system automatically generates a random Gaussian measurement matrix, which is normalized for stability.

## Algorithm Details

### L1 Minimization (LASSO)
Solves the optimization problem:
```
minimize ||Ax - y||₂² + λ||x||₁
```
where A is the measurement matrix, y is the measurements, and λ is the regularization parameter.

### Orthogonal Matching Pursuit (OMP)
Greedy algorithm that iteratively selects the best matching columns from the measurement matrix to build a sparse representation.

### Total Variation (TV)
Minimizes total variation to preserve edges while reconstructing the image:
```
minimize ||Ax - y||₂² + λ·TV(x)
```

## Performance

Typical reconstruction quality (32x32 spatial, 31 spectral bands, 30% compression):

| Algorithm | PSNR (dB) | Relative Error | Time (s) |
|-----------|-----------|----------------|----------|
| L1 Min    | ~25-30    | ~0.05-0.10     | ~5-10    |
| OMP       | ~20-25    | ~0.10-0.15     | ~1-2     |
| TV        | ~22-28    | ~0.08-0.12     | ~2-5     |

*Performance varies based on data sparsity and compression ratio.*

## API Reference

### CompressiveHSI

Main class for compressive hyperspectral imaging.

**Constructor:**
```python
CompressiveHSI(spatial_dims, spectral_bands, compression_ratio=0.5)
```

**Methods:**
- `compress(hyperspectral_cube)`: Compress HSI data
- `reconstruct_l1(measurements, max_iter=1000, tolerance=1e-6)`: Reconstruct using L1 minimization
- `reconstruct_omp(measurements, sparsity=None)`: Reconstruct using OMP
- `reconstruct_tval3(measurements, max_iter=100)`: Reconstruct using TV minimization

### Utility Functions

- `generate_synthetic_hsi(height, width, spectral_bands, sparse=True)`: Generate synthetic HSI data
- `compute_metrics(original, reconstructed)`: Compute reconstruction quality metrics

## Requirements

- Python >= 3.7
- NumPy >= 1.20.0
- SciPy >= 1.7.0

## License

This project is open source and available under the MIT License.

## References

1. Compressive Sensing: "Compressed Sensing" by D. Donoho (2006)
2. CASSI: "Coded Aperture Snapshot Spectral Imaging" by Gehm et al. (2007)
3. Total Variation: "Total Variation Minimization" by Rudin, Osher, and Fatemi (1992)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Karen Chou
