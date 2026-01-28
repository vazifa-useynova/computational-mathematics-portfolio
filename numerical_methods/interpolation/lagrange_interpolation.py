"""
Lagrange Polynomial Interpolation - Complete Implementation
==========================================================
Author: Vazifa Useynova | Baku Higher Oil School - Computer Engineering

Features:
1. Lagrange polynomial interpolation
2. Error analysis and Runge's phenomenon demonstration
3. Comparison with other interpolation methods
4. Real-world data fitting applications
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Callable
import time

class LagrangeInterpolator:
    """
    Implementation of Lagrange polynomial interpolation.
    """
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        """
        Initialize with data points.
        
        Parameters
        ----------
        x_data : np.ndarray
            X-coordinates of data points
        y_data : np.ndarray
            Y-coordinates of data points
        """
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must have same length")
        if len(np.unique(x_data)) != len(x_data):
            raise ValueError("x_data must have unique values")
        
        self.x_data = np.array(x_data, dtype=float)
        self.y_data = np.array(y_data, dtype=float)
        self.n = len(x_data)
        
    def basis_polynomial(self, k: int, x: float) -> float:
        """
        Calculate k-th Lagrange basis polynomial at point x.
        
        L_k(x) = Π_{j≠k} (x - x_j) / (x_k - x_j)
        """
        result = 1.0
        for j in range(self.n):
            if j != k:
                result *= (x - self.x_data[j]) / (self.x_data[k] - self.x_data[j])
        return result
    
    def interpolate(self, x: float) -> float:
        """
        Evaluate Lagrange polynomial at point x.
        
        P(x) = Σ_{k=0}^{n-1} y_k * L_k(x)
        """
        result = 0.0
        for k in range(self.n):
            result += self.y_data[k] * self.basis_polynomial(k, x)
        return result
    
    def interpolate_vector(self, x_values: np.ndarray) -> np.ndarray:
        """
        Evaluate polynomial at multiple points.
        """
        return np.array([self.interpolate(x) for x in x_values])
    
    def error_analysis(self, true_function: Callable, 
                      x_test: np.ndarray = None) -> Dict:
        """
        Analyze interpolation error compared to true function.
        """
        if x_test is None:
            x_test = np.linspace(min(self.x_data), max(self.x_data), 1000)
        
        y_true = true_function(x_test)
        y_interp = self.interpolate_vector(x_test)
        
        absolute_error = np.abs(y_interp - y_true)
        
        return {
            'max_error': np.max(absolute_error),
            'mean_error': np.mean(absolute_error),
            'x_test': x_test,
            'y_true': y_true,
            'y_interp': y_interp,
            'absolute_error': absolute_error
        }


class CubicSplineInterpolator:
    """
    Natural cubic spline interpolation for comparison.
    """
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x = np.array(x_data, dtype=float)
        self.y = np.array(y_data, dtype=float)
        self.n = len(x_data)
        self.coeffs = self._compute_coefficients()
    
    def _compute_coefficients(self) -> List[Tuple]:
        """
        Compute cubic spline coefficients using natural boundary conditions.
        """
        # Implementation of cubic spline algorithm
        n = self.n - 1
        
        # Step 1: Calculate h_i = x_{i+1} - x_i
        h = self.x[1:] - self.x[:-1]
        
        # Step 2: Set up tridiagonal system for second derivatives
        A = np.zeros((n+1, n+1))
        b = np.zeros(n+1)
        
        # Natural spline conditions
        A[0, 0] = 1
        A[n, n] = 1
        
        # Fill interior equations
        for i in range(1, n):
            A[i, i-1] = h[i-1]
            A[i, i] = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            b[i] = 3 * ((self.y[i+1] - self.y[i])/h[i] - 
                       (self.y[i] - self.y[i-1])/h[i-1])
        
        # Solve for second derivatives
        M = np.linalg.solve(A, b)
        
        # Compute coefficients for each segment
        coeffs = []
        for i in range(n):
            a = self.y[i]
            b_coeff = (self.y[i+1] - self.y[i])/h[i] - h[i]*(2*M[i] + M[i+1])/3
            c = M[i]
            d = (M[i+1] - M[i])/(3*h[i])
            coeffs.append((a, b_coeff, c, d))
        
        return coeffs
    
    def interpolate(self, x: float) -> float:
        """Evaluate cubic spline at point x."""
        # Find which segment x belongs to
        i = np.searchsorted(self.x, x) - 1
        i = max(0, min(i, len(self.coeffs)-1))
        
        a, b, c, d = self.coeffs[i]
        dx = x - self.x[i]
        return a + b*dx + c*dx**2 + d*dx**3


def runge_phenomenon_demo():
    """
    Demonstrate Runge's phenomenon with high-degree polynomial interpolation.
    """
    print("\n" + "="*70)
    print("RUNGE'S PHENOMENON DEMONSTRATION")
    print("="*70)
    
    # Runge's function: f(x) = 1/(1 + 25x²)
    def runge_function(x):
        return 1.0 / (1.0 + 25 * x**2)
    
    # Interpolation points
    degrees = [5, 10, 15]
    
    plt.figure(figsize=(15, 5))
    
    for idx, n in enumerate(degrees):
        # Equally spaced points
        x_points = np.linspace(-1, 1, n)
        y_points = runge_function(x_points)
        
        # Lagrange interpolation
        lagrange = LagrangeInterpolator(x_points, y_points)
        
        # Evaluation points
        x_eval = np.linspace(-1, 1, 1000)
        y_true = runge_function(x_eval)
        y_interp = lagrange.interpolate_vector(x_eval)
        
        # Calculate error
        error = np.abs(y_interp - y_true)
        
        # Plot
        plt.subplot(1, 3, idx + 1)
        plt.plot(x_eval, y_true, 'k-', linewidth=2, label='True function')
        plt.plot(x_eval, y_interp, 'r--', linewidth=1.5, label=f'Interpolation (n={n})')
        plt.scatter(x_points, y_points, color='blue', s=50, zorder=5, label='Data points')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'Runge Phenomenon (n={n})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        print(f"\nn = {n}:")
        print(f"  Max error: {np.max(error):.4f}")
        print(f"  Mean error: {np.mean(error):.4f}")
    
    plt.tight_layout()
    plt.savefig('runge_phenomenon.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to 'runge_phenomenon.png'")


def temperature_interpolation_example():
    """
    Real-world example: Temperature data interpolation.
    """
    print("\n" + "="*70)
    print("REAL-WORD APPLICATION: Temperature Data Interpolation")
    print("="*70)
    
    # Simulated temperature measurements (hourly)
    hours = np.array([0, 3, 6, 9, 12, 15, 18, 21])  # Time in hours
    temperatures = np.array([15.2, 13.5, 14.8, 19.3, 24.1, 25.8, 22.4, 18.9])  °C
    
    # Create interpolator
    temp_interpolator = LagrangeInterpolator(hours, temperatures)
    
    # Predict temperature at specific times
    prediction_times = [1, 4, 7, 10, 13, 16, 19, 22]
    predicted_temps = [temp_interpolator.interpolate(t) for t in prediction_times]
    
    print(f"\nMeasured Temperatures:")
    for h, t in zip(hours, temperatures):
        print(f"  {h:2d}:00 - {t:.1f}°C")
    
    print(f"\nInterpolated Predictions:")
    for h, t in zip(prediction_times, predicted_temps):
        print(f"  {h:2d}:00 - {t:.1f}°C")
    
    # Create smooth curve for visualization
    hours_smooth = np.linspace(0, 24, 1000)
    temps_smooth = temp_interpolator.interpolate_vector(hours_smooth)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(hours, temperatures, color='red', s=100, zorder=5, 
               label='Measured Data')
    plt.plot(hours_smooth, temps_smooth, 'b-', linewidth=2, alpha=0.7,
            label='Lagrange Interpolation')
    plt.xlabel('Time (hours)')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Interpolation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare with cubic spline
    spline = CubicSplineInterpolator(hours, temperatures)
    temps_spline = [spline.interpolate(h) for h in hours_smooth]
    
    plt.subplot(1, 2, 2)
    plt.plot(hours_smooth, temps_smooth, 'b-', linewidth=2, 
            label='Lagrange Polynomial')
    plt.plot(hours_smooth, temps_spline, 'g--', linewidth=2, 
            label='Cubic Spline')
    plt.scatter(hours, temperatures, color='red', s=50, zorder=5)
    plt.xlabel('Time (hours)')
    plt.ylabel('Temperature (°C)')
    plt.title('Comparison: Lagrange vs Cubic Spline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temperature_interpolation.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to 'temperature_interpolation.png'")


def convergence_study():
    """
    Study convergence of interpolation error with increasing nodes.
    """
    print("\n" + "="*70)
    print("CONVERGENCE ANALYSIS")
    print("="*70)
    
    # Test function: f(x) = sin(πx) + 0.5*cos(2πx)
    def test_function(x):
        return np.sin(np.pi * x) + 0.5 * np.cos(2 * np.pi * x)
    
    n_values = range(3, 21)
    errors_lagrange = []
    errors_spline = []
    
    for n in n_values:
        # Chebyshev nodes (better for polynomial interpolation)
        cheb_nodes = np.cos(np.pi * (2*np.arange(n) + 1) / (2*n))
        x_nodes = 0.5 * (cheb_nodes + 1)  # Map to [0, 1]
        y_nodes = test_function(x_nodes)
        
        # Lagrange interpolation
        lagrange = LagrangeInterpolator(x_nodes, y_nodes)
        
        # Evaluate error
        x_test = np.linspace(0, 1, 1000)
        y_true = test_function(x_test)
        y_lagrange = lagrange.interpolate_vector(x_test)
        error_lagrange = np.max(np.abs(y_lagrange - y_true))
        
        # Cubic spline interpolation
        spline = CubicSplineInterpolator(x_nodes, y_nodes)
        y_spline = np.array([spline.interpolate(x) for x in x_test])
        error_spline = np.max(np.abs(y_spline - y_true))
        
        errors_lagrange.append(error_lagrange)
        errors_spline.append(error_spline)
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(n_values, errors_lagrange, 'o-', linewidth=2, 
                label='Lagrange Interpolation')
    plt.semilogy(n_values, errors_spline, 's-', linewidth=2, 
                label='Cubic Spline')
    plt.xlabel('Number of Interpolation Nodes')
    plt.ylabel('Maximum Error')
    plt.title('Convergence of Interpolation Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add theoretical convergence rate
    theoretical = [1.0/n**4 for n in n_values]  # For smooth functions
    plt.semilogy(n_values, theoretical, 'k--', linewidth=1, 
                label='Theoretical O(n⁻⁴)', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('convergence_study.png', dpi=150, bbox_inches='tight')
    
    print("\nConvergence Analysis Results:")
    print("-" * 40)
    print("Nodes  Lagrange Error   Spline Error")
    print("-" * 40)
    for n, e_l, e_s in zip(n_values[-5:], errors_lagrange[-5:], errors_spline[-5:]):
        print(f"{n:5d}  {e_l:14.2e}  {e_s:14.2e}")
    
    print(f"\nConvergence plot saved to 'convergence_study.png'")


if __name__ == "__main__":
    print("="*70)
    print("LAGRANGE INTERPOLATION IMPLEMENTATION")
    print("Author: Vazifa Useynova | BHOS Computer Engineering")
    print("="*70)
    
    # Example 1: Simple polynomial interpolation
    print("\n1. Simple Polynomial Interpolation:")
    x_points = np.array([0, 1, 2, 3, 4])
    y_points = np.array([1, 3, 7, 13, 21])
    
    interpolator = LagrangeInterpolator(x_points, y_points)
    
    print(f"   Data points: x = {x_points}")
    print(f"               y = {y_points}")
    
    # Interpolate at specific points
    test_points = [0.5, 1.5, 2.5, 3.5]
    for x in test_points:
        y = interpolator.interpolate(x)
        print(f"   f({x}) = {y:.4f}")
    
    # Runge's phenomenon demonstration
    runge_phenomenon_demo()
    
    # Real-world application
    temperature_interpolation_example()
    
    # Convergence study
    convergence_study()
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("1. Lagrange interpolation provides exact fit at data points")
    print("2. Runge's phenomenon shows limitations with equally-spaced nodes")
    print("3. Chebyshev nodes provide better stability for high-degree polys")
    print("4. Cubic splines often more stable for real-world data")
    print("5. Error decreases with more nodes for smooth functions")
    print("="*70)
