import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps
import matplotlib.pyplot as plt
from plant import pendulum

# Simulation parameters
t_eval = np.linspace(0, 10, 1000)
z0 = [np.pi, 0.02]

# True dynamics
sol = solve_ivp(pendulum, (t_eval[0], t_eval[-1]), z0, t_eval=t_eval)
theta, omega = sol.y 

# Build state with extra features (what SINDy uses)
Z = np.vstack([
    theta,
    omega,
    np.sin(theta),
    np.cos(theta)
]).T  # shape: (1000, 4)

def lift_features(z):
    # z: shape (1, 4)
    theta, omega, s, c = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    return np.vstack([
        np.ones_like(theta), theta, omega, s, c,
        theta**2, theta*omega, theta*s, theta*c,
        omega**2, omega*s, omega*c, s**2, s*c, c**2
    ]).T  # shape: (1, 15)

# Fit SINDy model
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=1e-5),
    feature_names=['theta', 'omega', 'sin(theta)', 'cos(theta)']
)
model.fit(Z, t=t_eval)
model.print()

# Koopman matrix (learned with SINDy): 4x15
K = model.coefficients()

# Koopman dynamics in original state space (4D)
def koopman_rhs(t, z):
    z = z.reshape(1, -1)  # (1, 4)
    phi = lift_features(z)[0]  # (15,)
    dz = K @ phi  # (4,)
    return dz

# Initial condition (theta, omega, sin(theta), cos(theta))
theta0, omega0 = z0
Z0 = np.array([theta0, omega0, np.sin(theta0), np.cos(theta0)])

# Solve Koopman-predicted trajectory
koop_sol = solve_ivp(
    koopman_rhs, 
    (t_eval[0], t_eval[-1]), 
    Z0, 
    t_eval=t_eval, 
    method='RK45',
    max_step=0.01, 
    rtol=1e-6, atol=1e-8
)

# Extract Koopman estimates
Z_koop = koop_sol.y  # shape: (4, len(t_eval))

# def wrap_angle(theta):
#     return (theta + np.pi) % (2 * np.pi) - np.pi
def wrap_angle(theta):
    return theta % (2 * np.pi)
theta = wrap_angle(theta)
z_theta = wrap_angle(Z_koop[0,:])

# Plot theta
plt.plot(t_eval, theta, label='True θ')
plt.plot(t_eval, z_theta, '--', label='Koopman θ')
plt.xlabel('Time'); plt.ylabel('θ')
plt.legend(); plt.title('Theta: True vs Koopman')
plt.savefig("figures/koop_theta.png")
plt.show()

# Plot omega
plt.plot(t_eval, omega, label='True ω')
plt.plot(t_eval, Z_koop[1, :], '--', label='Koopman ω')
plt.xlabel('Time'); plt.ylabel('ω')
plt.legend(); plt.title('Omega: True vs Koopman')
plt.savefig("figures/koop_vel.png")
plt.show()
