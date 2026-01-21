# This script produces:
# (1) Re-projected ice mass graphs for NATO and US scenarios
# (2) Sensitivity analysis (Sobol-style variance contribution proxy)
# (3) Phase-space plots (Ice Mass vs dM/dt)
#
# Strictly analytical/statistical outputs only.

import numpy as np
import matplotlib.pyplot as plt

# Constants
rho_i = 917
L_f = 3.34e5
alpha_0 = 0.82
k_bc = 0.012
S_down = 200
M0 = 2.9e6

years = np.arange(2030, 2101)
N = 4000


# Core simulation

def simulate(C_bc_mean, C_bc_std, Q_min, Q_max, accel):
    M = np.zeros((N, len(years)))
    dM = np.zeros_like(M)
    M[:, 0] = M0

    for t in range(1, len(years)):
        C_bc = np.random.normal(C_bc_mean, C_bc_std, N)
        C_bc[C_bc < 0] = 0
        Q = np.random.uniform(Q_min, Q_max, N)

        alpha = alpha_0 - k_bc * C_bc
        alpha = np.clip(alpha, 0.6, alpha_0)

        S = ((1 - alpha) * S_down + Q) / (rho_i * L_f)
        D = accel * 0.6 * S

        dM[:, t] = -(S + D) * 1e4
        M[:, t] = M[:, t-1] + dM[:, t]

    return M, dM


# NATO scenario

M_nato, dM_nato = simulate(1.5, 0.3, 5, 8, 1.1)

mean_nato = M_nato.mean(axis=0)
p5_nato = np.percentile(M_nato, 5, axis=0)
p95_nato = np.percentile(M_nato, 95, axis=0)

plt.figure()
plt.plot(years, mean_nato)
plt.fill_between(years, p5_nato, p95_nato, alpha=0.3)
plt.xlabel("Year")
plt.ylabel("Ice Mass (Gt)")
plt.title("Ice Mass Projection – NATO Exercises")
plt.show()


# US occupancy scenario

M_us, dM_us = simulate(3.5, 0.5, 15, 22, 1.3)

mean_us = M_us.mean(axis=0)
p5_us = np.percentile(M_us, 5, axis=0)
p95_us = np.percentile(M_us, 95, axis=0)

plt.figure()
plt.plot(years, mean_us)
plt.fill_between(years, p5_us, p95_us, alpha=0.3)
plt.xlabel("Year")
plt.ylabel("Ice Mass (Gt)")
plt.title("Ice Mass Projection – US Occupancy")
plt.show()

# (3)Sensitivity analysis (variance contribution proxy)
final_mass = M_us[:, -1]

inputs = {
    "Black Carbon": np.random.normal(3.5, 0.5, N),
    "Anthropogenic Heat": np.random.uniform(15, 22, N),
    "Dynamic Acceleration": np.full(N, 1.3)
}

variances = []
labels = []

for key, val in inputs.items():
    variances.append(np.var(val))
    labels.append(key)

plt.figure()
plt.bar(labels, variances)
plt.ylabel("Variance Contribution (Proxy)")
plt.title("Sensitivity Analysis – Variance Drivers (US Scenario)")
plt.show()

# (4) Phase-space plot (M vs dM/dt)
mean_dM_us = dM_us.mean(axis=0)

plt.figure()
plt.plot(mean_us, mean_dM_us)
plt.xlabel("Ice Mass (Gt)")
plt.ylabel("dM/dt (Gt/year)")
plt.title("Phase-Space Trajectory – US Occupancy Scenario")
plt.show()
