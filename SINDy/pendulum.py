import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from plant import pendulum

## Modeling undamped unactuated single pendulum dynamics
# Define parameter
g=9.8
m = 1.0
l = 1.0
dt = 0.01
T = 5.0
t = np.arange(0,T,dt)

z0 = [np.pi, 0.02]
sol = solve_ivp(pendulum, [0,T], z0,t_eval=t)
theta_true, omega_true = sol.y
t_true = sol.t

data = np.stack((theta_true, omega_true), axis=-1)
model_Lib = ps.PolynomialLibrary(degree=2)+ps.FourierLibrary(n_frequencies=2)
optimizer = ps.STLSQ(threshold=0.1)
model = ps.SINDy(feature_library=model_Lib, feature_names=['theta', 'omega'], optimizer=optimizer)
model.fit(data, t=t_true)
model.print()

# Extrapolate for further time points
T_true = 10.0
t_test = np.arange(0,T_true,dt)
sim = model.simulate([np.pi/6, 0], t_test, integrator='solve_ivp')

# Compute true & plot
sol = solve_ivp(pendulum, [0,T_true], z0)
theta, omega = sol.y
t_true = sol.t

plt.plot(t_true,theta,label='True Angle')
plt.plot(t_test,sim[:,0],"--", label='SINDy Simulated Angle',linewidth=3)
plt.xlabel('Time [s]')
plt.ylabel('Position [rad]')
plt.title("True vs. SINDy Simulated Angle")
plt.legend()
plt.savefig("figures/pend_angle.png")
plt.show()

plt.plot(t_true,omega,label='True Angular Velocity')
plt.plot(t_test,sim[:,1],"--", label='SINDy Simulated Angular Velocity',linewidth=3)
plt.xlabel('Time [s]')
plt.ylabel('Position [rad/s]')
plt.title("True vs. SINDy Simulated Angular Velocity")
plt.legend()
plt.savefig("figures/pend_angle_vel.png")
plt.show()

