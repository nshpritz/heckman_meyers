import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, poisson
from scipy.integrate import trapezoid

# Data
x = np.array([3, 2, 4])
n = len(x)
sum_x = x.sum()

# Prior parameters (Gamma: shape α, rate β)
alpha, beta = 2, 1

# Posterior parameters
alpha_post = alpha + sum_x
beta_post = beta + n

# λ range for plotting
lam_vals = np.linspace(0, 6, 500)

# Prior PDF
prior_pdf = gamma.pdf(lam_vals, a=alpha, scale=1/beta)

# Likelihood (scaled for plotting)
likelihood_vals = np.exp(sum_x * np.log(lam_vals) - n * lam_vals)
likelihood_vals /= trapezoid(likelihood_vals, lam_vals)  # normalize

# Posterior PDF
posterior_pdf = gamma.pdf(lam_vals, a=alpha_post, scale=1/beta_post)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(lam_vals, prior_pdf, 'b-', lw=2, label='Prior (Gamma)')
plt.plot(lam_vals, likelihood_vals, 'g--', lw=2, label='Likelihood (scaled)')
plt.plot(lam_vals, posterior_pdf, 'r-', lw=2, label='Posterior (Gamma)')

plt.axvline(alpha / beta, color='b', linestyle=':', 
            label=f'Prior mean = {alpha / beta:.2f}')

plt.axvline(alpha_post / beta_post, color='r', linestyle=':', 
            label=f'Posterior mean = {alpha_post / beta_post:.2f}')

plt.title("Bayesian Update for Poisson Rate λ")
plt.xlabel("λ")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
