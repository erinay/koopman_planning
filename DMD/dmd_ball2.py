#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simulate and reconstruct the dynamics of a ball thrown toward the observer using DMD.
"""

import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD, BOPDMD
from pydmd.preprocessing import hankel_preprocessing

# ----------------------------
# Simulate Ball Trajectory
# ----------------------------

g = 9.81  # gravity (m/s^2)
dt = 0.05  # time step
T = 3.0    # total time (s)
t = np.arange(0, T, dt)

# Initial conditions: ball is thrown toward the observer
x0, z0 = 0.0, 10.0       # lateral and depth (z is toward observer)
vx0, vz0 = 1.0, -5.0     # initial velocity

# Simulate position and velocity
x = x0 + vx0 * t
z = z0 + vz0 * t - 0.5 * g * t**2
vx = vx0 * np.ones_like(t)
vz = vz0 - g * t

# State matrix: each row is a variable, each column is a time snapshot
X = np.vstack([x, z, vx, vz])

# Optionally add noise
noise_level = 0.5
X_noisy = X + noise_level * np.random.randn(*X.shape)

# ----------------------------
# Apply Standard DMD
# ----------------------------

dmd = DMD(svd_rank=4)
dmd.fit(X_noisy)
X_dmd = dmd.reconstructed_data.real

# ----------------------------
# Apply BOP-DMD with Delay Embedding
# ----------------------------

d = 2  # number of delays
bopdmd = BOPDMD(svd_rank=4, num_trials=0, eig_constraints={"conjugate_pairs"})
delay_bop = hankel_preprocessing(bopdmd, d=d)
delay_bop.fit(X_noisy, t=t[: -d + 1])
X_bop = delay_bop.reconstructed_data.real
t_bop = t[: -d + 1]

# ----------------------------
# Plotting
# ----------------------------

plt.figure(figsize=(12, 6))

# Position comparison
plt.subplot(2, 1, 1)
plt.plot(t, x, label='True x')
plt.plot(t, z, label='True z')
plt.plot(t, X_dmd[0], '--', label='DMD x')
plt.plot(t, X_dmd[1], '--', label='DMD z')
plt.plot(t_bop, X_bop[0, :len(t_bop)], ':', label='BOP-DMD x')
plt.plot(t_bop, X_bop[1, :len(t_bop)], ':', label='BOP-DMD z')
plt.title("Ball Position (Toward Observer)")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend()

# Velocity comparison
plt.subplot(2, 1, 2)
plt.plot(t, vx, label='True vx')
plt.plot(t, vz, label='True vz')
plt.plot(t, X_dmd[2], '--', label='DMD vx')
plt.plot(t, X_dmd[3], '--', label='DMD vz')
plt.plot(t_bop, X_bop[2, :len(t_bop)], ':', label='BOP-DMD vx')
plt.plot(t_bop, X_bop[3, :len(t_bop)], ':', label='BOP-DMD vz')
plt.title("Ball Velocity (Toward Observer)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.legend()

plt.tight_layout()
plt.show()

# ----------------------------
# Print Eigenvalues
# ----------------------------

print("\nStandard DMD eigenvalues (frequencies):")
print(np.round(np.log(dmd.eigs) / dt, 3))

print("\nBOP-DMD eigenvalues (frequencies):")
print(np.round(np.log(delay_bop.eigs) / dt, 3))
