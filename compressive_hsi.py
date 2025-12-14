"""
Compressive Hyperspectral Imaging (CHI) Implementation

This module implements compressive sensing techniques for hyperspectral imaging,
including measurement matrix generation, compression, and reconstruction algorithms.
"""

import numpy as np
from typing import Tuple, Optional


class CompressiveHSI:
    """
    Compressive Hyperspectral Imaging class implementing coded aperture
    snapshot spectral imaging (CASSI) with reconstruction algorithms.
    """
    
    def __init__(self, spatial_dims: Tuple[int, int], spectral_bands: int, 
                 compression_ratio: float = 0.5):
        """
        Initialize the Compressive HSI system.
        
        Args:
            spatial_dims: Tuple of (height, width) for spatial dimensions
            spectral_bands: Number of spectral bands
            compression_ratio: Ratio of measurements to original data size (0-1)
        """
        self.height, self.width = spatial_dims
        self.spectral_bands = spectral_bands
        self.compression_ratio = compression_ratio
        
        # Calculate number of measurements
        total_size = self.height * self.width * self.spectral_bands
        self.num_measurements = int(total_size * compression_ratio)
        
        # Generate measurement matrix
        self.measurement_matrix = self._generate_measurement_matrix()
        
    def _generate_measurement_matrix(self) -> np.ndarray:
        """
        Generate random Gaussian measurement matrix.
        
        Returns:
            Measurement matrix of shape (num_measurements, total_size)
        """
        total_size = self.height * self.width * self.spectral_bands
        matrix = np.random.randn(self.num_measurements, total_size)
        
        # Normalize rows
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix
    
    def compress(self, hyperspectral_cube: np.ndarray) -> np.ndarray:
        """
        Compress hyperspectral data using the measurement matrix.
        
        Args:
            hyperspectral_cube: Hyperspectral data of shape (height, width, spectral_bands)
            
        Returns:
            Compressed measurements vector
        """
        # Flatten the hyperspectral cube
        flat_data = hyperspectral_cube.reshape(-1)
        
        # Apply measurement matrix
        measurements = self.measurement_matrix @ flat_data
        return measurements
    
    def reconstruct_l1(self, measurements: np.ndarray, 
                       max_iter: int = 1000, 
                       tolerance: float = 1e-6,
                       lambda_l1: float = 0.01) -> np.ndarray:
        """
        Reconstruct hyperspectral cube using L1 minimization (ISTA - Iterative Soft Thresholding).
        
        Args:
            measurements: Compressed measurements
            max_iter: Maximum iterations for optimization
            tolerance: Convergence tolerance
            lambda_l1: L1 regularization parameter
            
        Returns:
            Reconstructed hyperspectral cube of shape (height, width, spectral_bands)
        """
        total_size = self.height * self.width * self.spectral_bands
        
        # Initialize with pseudo-inverse
        x = np.linalg.lstsq(self.measurement_matrix, measurements, rcond=None)[0]
        
        # Precompute step size (using power iteration estimate)
        L = np.linalg.norm(self.measurement_matrix, ord=2) ** 2
        step_size = 1.0 / L
        
        # ISTA iterations
        for i in range(max_iter):
            x_old = x.copy()
            
            # Gradient descent step
            residual = self.measurement_matrix @ x - measurements
            gradient = self.measurement_matrix.T @ residual
            x = x - step_size * gradient
            
            # Soft thresholding (L1 proximal operator)
            threshold = lambda_l1 * step_size
            x = np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
            
            # Check convergence
            if np.linalg.norm(x - x_old) / (np.linalg.norm(x_old) + 1e-10) < tolerance:
                break
        
        # Reshape to original dimensions
        reconstructed = x.reshape(self.height, self.width, self.spectral_bands)
        return reconstructed
    
    def reconstruct_omp(self, measurements: np.ndarray, 
                        sparsity: int = None) -> np.ndarray:
        """
        Reconstruct using Orthogonal Matching Pursuit (OMP).
        
        Args:
            measurements: Compressed measurements
            sparsity: Expected sparsity level (number of non-zero coefficients)
            
        Returns:
            Reconstructed hyperspectral cube
        """
        if sparsity is None:
            sparsity = self.num_measurements // 4
        
        total_size = self.height * self.width * self.spectral_bands
        x = np.zeros(total_size)
        residual = measurements.copy()
        indices = []
        
        for _ in range(sparsity):
            # Find best matching column
            correlations = np.abs(self.measurement_matrix.T @ residual)
            best_idx = np.argmax(correlations)
            indices.append(best_idx)
            
            # Update solution using least squares on selected columns
            A_selected = self.measurement_matrix[:, indices]
            x_selected = np.linalg.lstsq(A_selected, measurements, rcond=None)[0]
            
            # Update residual
            residual = measurements - A_selected @ x_selected
            
            # Check convergence
            if np.linalg.norm(residual) < 1e-6:
                break
        
        # Set solution
        x[indices] = x_selected
        
        # Reshape to original dimensions
        reconstructed = x.reshape(self.height, self.width, self.spectral_bands)
        return reconstructed
    
    def reconstruct_tval3(self, measurements: np.ndarray, 
                          max_iter: int = 100) -> np.ndarray:
        """
        Reconstruct using Total Variation minimization (simplified version).
        
        Args:
            measurements: Compressed measurements
            max_iter: Maximum iterations
            
        Returns:
            Reconstructed hyperspectral cube
        """
        total_size = self.height * self.width * self.spectral_bands
        
        # Initialize with pseudo-inverse solution
        x = np.linalg.lstsq(self.measurement_matrix, measurements, rcond=None)[0]
        
        # Iterative refinement with TV regularization
        lambda_tv = 0.01
        step_size = 0.1
        
        for _ in range(max_iter):
            # Data fidelity gradient
            data_grad = self.measurement_matrix.T @ (self.measurement_matrix @ x - measurements)
            
            # TV regularization (simplified - use gradient magnitude)
            x_reshaped = x.reshape(self.height, self.width, self.spectral_bands)
            tv_grad = np.zeros_like(x_reshaped)
            
            # Spatial TV gradients
            tv_grad[:-1, :, :] += x_reshaped[:-1, :, :] - x_reshaped[1:, :, :]
            tv_grad[:, :-1, :] += x_reshaped[:, :-1, :] - x_reshaped[:, 1:, :]
            tv_grad = tv_grad.reshape(-1)
            
            # Update
            x = x - step_size * (data_grad + lambda_tv * tv_grad)
        
        # Reshape to original dimensions
        reconstructed = x.reshape(self.height, self.width, self.spectral_bands)
        return reconstructed


def generate_synthetic_hsi(height: int, width: int, spectral_bands: int, 
                          sparse: bool = True) -> np.ndarray:
    """
    Generate synthetic hyperspectral image for testing.
    
    Args:
        height: Image height
        width: Image width
        spectral_bands: Number of spectral bands
        sparse: If True, generate sparse synthetic data
        
    Returns:
        Synthetic hyperspectral cube of shape (height, width, spectral_bands)
    """
    cube = np.zeros((height, width, spectral_bands))
    
    if sparse:
        # Create sparse spectral signatures
        num_materials = 5
        for i in range(num_materials):
            # Random spatial region
            y_start = np.random.randint(0, height - height // 4)
            x_start = np.random.randint(0, width - width // 4)
            y_end = y_start + np.random.randint(height // 8, height // 4)
            x_end = x_start + np.random.randint(width // 8, width // 4)
            
            # Random spectral signature (sparse)
            signature = np.zeros(spectral_bands)
            num_peaks = np.random.randint(1, 4)
            peak_indices = np.random.choice(spectral_bands, num_peaks, replace=False)
            signature[peak_indices] = np.random.rand(num_peaks)
            
            cube[y_start:y_end, x_start:x_end, :] += signature
    else:
        # Dense synthetic data
        cube = np.random.rand(height, width, spectral_bands)
    
    return cube


def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """
    Compute reconstruction quality metrics.
    
    Args:
        original: Original hyperspectral cube
        reconstructed: Reconstructed hyperspectral cube
        
    Returns:
        Dictionary with PSNR, RMSE, and relative error
    """
    # Ensure same shape
    assert original.shape == reconstructed.shape
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((original - reconstructed) ** 2))
    
    # Peak Signal-to-Noise Ratio
    max_val = np.max(original)
    if max_val > 0:
        psnr = 20 * np.log10(max_val / rmse) if rmse > 0 else float('inf')
    else:
        psnr = float('inf')
    
    # Relative error
    rel_error = np.linalg.norm(original - reconstructed) / np.linalg.norm(original)
    
    return {
        'RMSE': rmse,
        'PSNR': psnr,
        'Relative Error': rel_error
    }
