"""
Example usage of Compressive Hyperspectral Imaging

This script demonstrates the compression and reconstruction of hyperspectral images
using different reconstruction algorithms.
"""

import numpy as np
import time
from compressive_hsi import CompressiveHSI, generate_synthetic_hsi, compute_metrics


def main():
    """Main demonstration of compressive hyperspectral imaging."""
    
    print("=" * 70)
    print("Compressive Hyperspectral Imaging Demo")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    height, width = 32, 32
    spectral_bands = 31
    compression_ratio = 0.3
    
    print(f"\nConfiguration:")
    print(f"  Spatial dimensions: {height} x {width}")
    print(f"  Spectral bands: {spectral_bands}")
    print(f"  Compression ratio: {compression_ratio}")
    print(f"  Original data size: {height * width * spectral_bands}")
    
    # Generate synthetic hyperspectral image
    print("\n" + "-" * 70)
    print("Generating synthetic hyperspectral image...")
    hsi_cube = generate_synthetic_hsi(height, width, spectral_bands, sparse=True)
    print(f"Generated cube shape: {hsi_cube.shape}")
    print(f"Data range: [{hsi_cube.min():.4f}, {hsi_cube.max():.4f}]")
    
    # Initialize compressive sensing system
    print("\n" + "-" * 70)
    print("Initializing Compressive HSI system...")
    chi_system = CompressiveHSI((height, width), spectral_bands, compression_ratio)
    print(f"Number of measurements: {chi_system.num_measurements}")
    print(f"Compression: {height * width * spectral_bands} -> {chi_system.num_measurements}")
    print(f"Compression factor: {(height * width * spectral_bands) / chi_system.num_measurements:.2f}x")
    
    # Compress the hyperspectral cube
    print("\n" + "-" * 70)
    print("Compressing hyperspectral data...")
    start_time = time.time()
    measurements = chi_system.compress(hsi_cube)
    compress_time = time.time() - start_time
    print(f"Compression completed in {compress_time:.4f} seconds")
    print(f"Measurements shape: {measurements.shape}")
    
    # Test different reconstruction algorithms
    algorithms = [
        ("L1 Minimization (ISTA)", "l1", {"max_iter": 200}),
        ("Orthogonal Matching Pursuit (OMP)", "omp", {"sparsity": 100}),
        ("Total Variation (TV)", "tval3", {"max_iter": 50})
    ]
    
    print("\n" + "=" * 70)
    print("Reconstruction Results")
    print("=" * 70)
    
    results = []
    
    for algo_name, algo_type, algo_params in algorithms:
        print(f"\n{algo_name}:")
        print("-" * 70)
        
        # Reconstruct
        start_time = time.time()
        if algo_type == "l1":
            reconstructed = chi_system.reconstruct_l1(measurements, **algo_params)
        elif algo_type == "omp":
            reconstructed = chi_system.reconstruct_omp(measurements, **algo_params)
        elif algo_type == "tval3":
            reconstructed = chi_system.reconstruct_tval3(measurements, **algo_params)
        
        recon_time = time.time() - start_time
        
        # Compute metrics
        metrics = compute_metrics(hsi_cube, reconstructed)
        
        print(f"  Reconstruction time: {recon_time:.4f} seconds")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  PSNR: {metrics['PSNR']:.2f} dB")
        print(f"  Relative Error: {metrics['Relative Error']:.6f}")
        
        results.append({
            'algorithm': algo_name,
            'time': recon_time,
            'metrics': metrics
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Algorithm':<40} {'Time (s)':<12} {'PSNR (dB)':<12} {'Rel Error':<12}")
    print("-" * 70)
    for result in results:
        print(f"{result['algorithm']:<40} "
              f"{result['time']:<12.4f} "
              f"{result['metrics']['PSNR']:<12.2f} "
              f"{result['metrics']['Relative Error']:<12.6f}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
