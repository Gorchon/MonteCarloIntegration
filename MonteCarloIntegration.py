import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def monte_carlo_integration(func, a, b, num_samples):
    """
    Estimates the integral of `func` from `a` to `b` using Monte Carlo integration.

    Parameters:
    - func: The function to integrate.
    - a: Lower limit of integration.
    - b: Upper limit of integration.
    - num_samples: Number of random samples to use.

    Returns:
    - estimate: The estimated value of the integral.
    """
    samples = np.random.uniform(a, b, num_samples)
    evaluations = func(samples)
    estimate = (b - a) * np.mean(evaluations)
    return estimate

def plot_convergence(func, a, b, exact_value=None, max_samples=100000):
    """
    Plots the convergence of the Monte Carlo integration estimate.

    Parameters:
    - func: The function to integrate.
    - a: Lower limit of integration.
    - b: Upper limit of integration.
    - exact_value: Exact value of the integral (if known).
    - max_samples: Maximum number of samples to use.
    """
    sample_sizes = np.logspace(2, np.log10(max_samples), num=100, dtype=int)
    estimates = []
    errors = []
    for n in sample_sizes:
        estimate = monte_carlo_integration(func, a, b, n)
        estimates.append(estimate)
        if exact_value is not None:
            errors.append(abs(estimate - exact_value))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(sample_sizes, estimates, label='Monte Carlo Estimate')
    if exact_value is not None:
        plt.hlines(exact_value, sample_sizes[0], sample_sizes[-1], colors='r', linestyles='dashed', label='Exact Value')
    plt.xscale('log')
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Integral Estimate')
    plt.title('Convergence of Monte Carlo Integration')
    plt.legend()
    plt.grid(True)

    if exact_value is not None:
        plt.subplot(1, 2, 2)
        plt.plot(sample_sizes, errors, label='Error')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of Samples (log scale)')
        plt.ylabel('Absolute Error (log scale)')
        plt.title('Error Decrease Over Samples')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

# Example 1: Integrate f(x) = x^2 from 0 to 1
def func1(x):
    return x ** 2

a1, b1 = 0, 1
exact_value1 = (b1 ** 3 - a1 ** 3) / 3  # Analytical integral of x^2

estimate1 = monte_carlo_integration(func1, a1, b1, 100000)
print(f"Monte Carlo estimate of integral of x^2 from {a1} to {b1}: {estimate1}")
print(f"Exact value: {exact_value1}\n")

plot_convergence(func1, a1, b1, exact_value=exact_value1)

# Example 2: Integrate f(x) = sin(x) from 0 to π
def func2(x):
    return np.sin(x)

a2, b2 = 0, np.pi
exact_value2 = 2  # Analytical integral of sin(x) from 0 to π

estimate2 = monte_carlo_integration(func2, a2, b2, 100000)
print(f"Monte Carlo estimate of integral of sin(x) from {a2} to {b2}: {estimate2}")
print(f"Exact value: {exact_value2}\n")

plot_convergence(func2, a2, b2, exact_value=exact_value2)

# Example 3: Integrate f(x) = e^{-x^2} from -1 to 1
def func3(x):
    return np.exp(-x ** 2)

a3, b3 = -1, 1
exact_value3 = erf(1) * np.sqrt(np.pi)

estimate3 = monte_carlo_integration(func3, a3, b3, 100000)
print(f"Monte Carlo estimate of integral of exp(-x^2) from {a3} to {b3}: {estimate3}")
print(f"Exact value: {exact_value3}\n")

plot_convergence(func3, a3, b3, exact_value=exact_value3)
