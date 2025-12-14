"""
Quick test script for Compressive Hyperspectral Imaging

This is a simplified test that runs faster for quick validation.
"""

import numpy as np
from compressive_hsi import CompressiveHSI, generate_synthetic_hsi, compute_metrics


def main():
    """Quick test of compressive hyperspectral imaging."""
    
    print("Quick Test of Compressive Hyperspectral Imaging")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Smaller parameters for faster testing
    height, width = 16, 16
    spectral_bands = 15
    compression_ratio = 0.5
    
    print(f"\nTest Configuration:")
    print(f"  Spatial: {height}x{width}, Spectral bands: {spectral_bands}")
    print(f"  Compression ratio: {compression_ratio}")
    
    # Generate synthetic data
    print("\nGenerating synthetic hyperspectral image...")
    hsi_cube = generate_synthetic_hsi(height, width, spectral_bands, sparse=True)
    print(f"  Cube shape: {hsi_cube.shape}")
    
    # Initialize system
    print("\nInitializing Compressive HSI system...")
    chi_system = CompressiveHSI((height, width), spectral_bands, compression_ratio)
    print(f"  Compression: {height*width*spectral_bands} -> {chi_system.num_measurements}")
    
    # Compress
    print("\nCompressing...")
    measurements = chi_system.compress(hsi_cube)
    print(f"  Done. Measurements: {measurements.shape}")
    
    # Test OMP (fastest algorithm)
    print("\nTesting OMP Reconstruction...")
    reconstructed = chi_system.reconstruct_omp(measurements, sparsity=50)
    metrics = compute_metrics(hsi_cube, reconstructed)
    
    print(f"\nResults:")
    print(f"  PSNR: {metrics['PSNR']:.2f} dB")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  Relative Error: {metrics['Relative Error']:.6f}")
    
    # Validate reconstruction quality
    if metrics['PSNR'] > 15:  # Reasonable PSNR threshold
        print("\n✓ Test PASSED - Reconstruction quality is good!")
        return 0
    else:
        print("\n✗ Test FAILED - Reconstruction quality is poor!")
        return 1


if __name__ == "__main__":
    exit(main())
