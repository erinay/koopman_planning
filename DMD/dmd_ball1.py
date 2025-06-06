"""
Estimate the dynamics of a falling ball using Dynamic Mode Decomposition (DMD).
"""

import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD, BOPDMD
from pydmd.preprocessing import hankel_preprocessing

# ----------------------------
# Simulate 1D falling ball
# ----------------------------

g = 9.81  # gravity (m/s^2)
dt = 0.05  # time step (s)
T = 5.0  # total duration (s)
t = np.arange(0, T, dt)

# Initial conditions
x0 = 10.0  # initial position (m)
v0 = 0.0   # initial velocity (m/s)

# Position and velocity over time
x = x0 + v0 * t - 0.5 * g * t**2
v = v0 - g * t

# Stack into state matrix: each column is a snapshot at one time step
X = np.vstack((x, v))

# Optionally add noise
noise_level = 0.05
X_noisy = X + noise_level * np.random.randn(*X.shape)

# ----------------------------
# Standard DMD
# ----------------------------

dmd = DMD(svd_rank=2)
dmd.fit(X_noisy)

# DMD reconstruction
X_dmd = dmd.reconstructed_data.real

# ----------------------------
# BOP-DMD with delay embedding
# ----------------------------

d = 2  # number of delays
bopdmd = BOPDMD(svd_rank=2, num_trials=0, eig_constraints={"conjugate_pairs"})
delay_bopdmd = hankel_preprocessing(bopdmd, d=d)
delay_bopdmd.fit(X_noisy, t=t[: -d + 1])
X_bop = delay_bopdmd.reconstructed_data.real

# ----------------------------
# Plotting
# ----------------------------

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, x, label="True x")
plt.plot(t, v, label="True v")
plt.plot(t, X_dmd[0, :], "--", label="DMD x")
plt.plot(t, X_dmd[1, :], "--", label="DMD v")
plt.title("Standard DMD Reconstruction")
plt.xlabel("Time (s)")
plt.ylabel("State")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, x, label="True x")
plt.plot(t, v, label="True v")
plt.plot(t[: -d + 1], X_bop[0, :len(t[: -d + 1])], "--", label="BOP-DMD x")
plt.plot(t[: -d + 1], X_bop[1, :len(t[: -d + 1])], "--", label="BOP-DMD v")
plt.title("BOP-DMD (Delayed) Reconstruction")
plt.xlabel("Time (s)")
plt.ylabel("State")
plt.legend()

plt.tight_layout()
plt.show()

# ----------------------------
# Print DMD Frequencies
# ----------------------------

print("\nStandard DMD eigenvalues:")
print(np.round(np.log(dmd.eigs) / dt, 3))

print("\nBOP-DMD eigenvalues (imag part only):")
print(np.round(np.imag(np.log(delay_bopdmd.eigs)) / dt, 3))
