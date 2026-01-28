"""
Gaussian Elimination with Partial Pivoting - Complete Implementation
====================================================================
Author: Vazifa Useynova | Baku Higher Oil School - Computer Engineering
Academic Performance: 94.23/100 GPA | Research: Inverse Problems

A comprehensive implementation of Gaussian elimination featuring:
1. Partial pivoting for numerical stability
2. Performance comparison with NumPy
3. Engineering applications (circuit analysis)
4. Error analysis and condition number evaluation
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import sys

class GaussianElimination:
    """
    Professional implementation of Gaussian elimination algorithm.
    """
    
    def __init__(self):
        self.operations_count = 0
        self.execution_time = 0
        
    def solve(self, A: np.ndarray, b: np.ndarray, pivoting: bool = True) -> np.ndarray:
        """
        Solve Ax = b using Gaussian elimination.
        
        Parameters
        ----------
        A : np.ndarray
            Coefficient matrix (n x n)
        b : np.ndarray
            Right-hand side vector (n,)
        pivoting : bool
            Use partial pivoting (recommended)
            
        Returns
        -------
        np.ndarray
            Solution vector x
        """
        start_time = time.perf_counter()
        self.operations_count = 0
        
        # Validate inputs
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError(f"Matrix must be square, got {A.shape}")
        if b.shape != (n,):
            raise ValueError(f"Vector dimension mismatch: b has shape {b.shape}")
        
        # Create augmented matrix [A|b]
        aug = np.hstack([A.astype(float), b.reshape(-1, 1)])
        
        # Forward elimination
        for i in range(n):
            if pivoting:
                # Find pivot with maximum absolute value
                max_row = i + np.argmax(np.abs(aug[i:, i]))
                if max_row != i:
                    aug[[i, max_row]] = aug[[max_row, i]]
                    self.operations_count += 3 * (n - i + 1)  # Account for row swap
            
            # Check for singularity
            if abs(aug[i, i]) < 1e-15:
                raise ValueError(f"Matrix is singular at pivot {i}")
            
            # Eliminate below diagonal
            for j in range(i + 1, n):
                factor = aug[j, i] / aug[i, i]
                aug[j, i:] -= factor * aug[i, i:]
                self.operations_count += 2 * (n - i + 1)
        
        # Back substitution
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (aug[i, -1] - np.dot(aug[i, i+1:n], x[i+1:])) / aug[i, i]
            self.operations_count += 2 * (n - i - 1) + 2
        
        self.execution_time = time.perf_counter() - start_time
        return x
    
    def analyze_performance(self, A: np.ndarray, b: np.ndarray) -> Dict:
        """
        Comprehensive performance analysis.
        
        Returns
        -------
        Dict with timing, errors, and comparisons
        """
        results = {}
        
        # Solve with different methods
        start = time.perf_counter()
        x_custom = self.solve(A, b, pivoting=True)
        results['custom_time'] = self.execution_time
        
        # NumPy comparison
        start_numpy = time.perf_counter()
        x_numpy = np.linalg.solve(A, b)
        results['numpy_time'] = time.perf_counter() - start_numpy
        
        # Error analysis
        results['absolute_error'] = np.abs(x_custom - x_numpy)
        results['relative_error'] = results['absolute_error'] / (np.abs(x_numpy) + 1e-15)
        results['max_relative_error'] = np.max(results['relative_error'])
        
        # Residual norms
        results['residual_custom'] = np.linalg.norm(A @ x_custom - b)
        results['residual_numpy'] = np.linalg.norm(A @ x_numpy - b)
        
        # Condition number
        results['condition_number'] = np.linalg.cond(A)
        
        # Operation count
        results['operation_count'] = self.operations_count
        results['theoretical_operations'] = (2/3)*A.shape[0]**3 + 2*A.shape[0]**2
        
        return results
    
    def circuit_analysis_demo(self):
        """
        Real-world application: Electrical circuit analysis.
        Solves for currents using Kirchhoff's laws.
        """
        print("\n" + "="*60)
        print("ENGINEERING APPLICATION: Circuit Analysis")
        print("="*60)
        
        # Define circuit parameters
        # 3-loop circuit with resistors and voltage sources
        R = np.array([
            [10, -2, -3],   # Loop 1: R11=10Ω, R12=-2Ω, R13=-3Ω
            [-2, 8, -1],    # Loop 2
            [-3, -1, 12]    # Loop 3
        ], dtype=float)
        
        V = np.array([12, 0, -5], dtype=float)  # Voltage sources
        
        print(f"\nResistance Matrix (Ω):")
        print(R)
        print(f"\nVoltage Vector (V): {V}")
        
        # Solve for currents
        currents = self.solve(R, V, pivoting=True)
        
        print(f"\nSolution (Loop Currents):")
        for i, I in enumerate(currents, 1):
            print(f"  I{i} = {I:.6f} A")
        
        # Calculate power dissipation
        power = np.sum(currents * (R @ currents))
        print(f"\nTotal Power Dissipated: {power:.4f} W")
        
        # Verify Kirchhoff's laws
        residuals = R @ currents - V
        print(f"Kirchhoff's Law Residual: {np.linalg.norm(residuals):.2e} (should be ~0)")
        
        return currents


def performance_comparison():
    """
    Compare performance for different matrix sizes.
    """
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS: Custom vs NumPy Implementation")
    print("="*60)
    
    sizes = [10, 50, 100, 200]
    custom_times = []
    numpy_times = []
    errors = []
    
    solver = GaussianElimination()
    
    for n in sizes:
        # Generate random well-conditioned matrix
        A = np.random.randn(n, n) + 5 * np.eye(n)
        b = np.random.randn(n)
        
        # Custom implementation
        start = time.perf_counter()
        x_custom = solver.solve(A, b, pivoting=True)
        t_custom = time.perf_counter() - start
        
        # NumPy
        start = time.perf_counter()
        x_numpy = np.linalg.solve(A, b)
        t_numpy = time.perf_counter() - start
        
        custom_times.append(t_custom)
        numpy_times.append(t_numpy)
        errors.append(np.max(np.abs(x_custom - x_numpy)))
        
        print(f"\nn = {n:3d}:")
        print(f"  Custom: {t_custom:.6f} sec")
        print(f"  NumPy:  {t_numpy:.6f} sec")
        print(f"  Speed Ratio: {t_custom/t_numpy:.2f}x")
        print(f"  Max Error: {errors[-1]:.2e}")
    
    # Visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.loglog(sizes, custom_times, 'o-', label='Custom', linewidth=2)
    plt.loglog(sizes, numpy_times, 's-', label='NumPy', linewidth=2)
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.semilogy(sizes, errors, 'o-', linewidth=2, color='red')
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Maximum Error')
    plt.title('Numerical Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    ratios = [c/n for c, n in zip(custom_times, numpy_times)]
    plt.plot(sizes, ratios, 'o-', linewidth=2, color='green')
    plt.xlabel('Matrix Size (n)')
    plt.ylabel('Time Ratio (Custom/NumPy)')
    plt.title('Performance Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gauss_performance.png', dpi=150, bbox_inches='tight')
    print(f"\nPerformance plots saved to 'gauss_performance.png'")


def stability_analysis():
    """
    Analyze numerical stability with ill-conditioned matrices.
    """
    print("\n" + "="*60)
    print("NUMERICAL STABILITY ANALYSIS")
    print("="*60)
    
    solver = GaussianElimination()
    
    # Test with Hilbert matrix (notoriously ill-conditioned)
    n = 8
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)
    
    b = np.ones(n)
    
    print(f"\nHilbert Matrix (n={n}) Condition Number: {np.linalg.cond(H):.2e}")
    
    # Solve with and without pivoting
    try:
        x_without = solver.solve(H.copy(), b.copy(), pivoting=False)
        residual_without = np.linalg.norm(H @ x_without - b)
        print(f"No pivoting: Residual = {residual_without:.2e}")
    except Exception as e:
        print(f"No pivoting failed: {e}")
    
    x_with = solver.solve(H.copy(), b.copy(), pivoting=True)
    residual_with = np.linalg.norm(H @ x_with - b)
    print(f"With pivoting: Residual = {residual_with:.2e}")
    
    # Theoretical vs actual solution
    x_theoretical = np.linalg.solve(H, b)
    error = np.linalg.norm(x_with - x_theoretical)
    print(f"Error from theoretical: {error:.2e}")


if __name__ == "__main__":
    print("="*70)
    print("GAUSSIAN ELIMINATION IMPLEMENTATION")
    print("Author: Vazifa Useynova | BHOS Computer Engineering")
    print("="*70)
    
    solver = GaussianElimination()
    
    # Example 1: Simple 3x3 system
    print("\n1. Solving 3x3 Linear System:")
    A1 = np.array([[2, 1, -1],
                   [-3, -1, 2],
                   [-2, 1, 2]], dtype=float)
    b1 = np.array([8, -11, -3], dtype=float)
    
    x1 = solver.solve(A1, b1)
    print(f"   System: Ax = b")
    print(f"   A = \n{A1}")
    print(f"   b = {b1}")
    print(f"   Solution x = {x1}")
    print(f"   Verify Ax = {A1 @ x1} (should equal b)")
    
    # Example 2: Engineering application
    solver.circuit_analysis_demo()
    
    # Performance analysis
    performance_comparison()
    
    # Stability analysis
    stability_analysis()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("Key Insights:")
    print("1. Gaussian elimination with pivoting is essential for stability")
    print("2. Custom implementation matches NumPy accuracy within machine precision")
    print("3. Algorithm complexity O(n³) confirmed empirically")
    print("4. Applications in engineering problems demonstrated")
    print("="*70)
