## Starting with falling ball example
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

t = np.linspace(0,2,200)
g = -9.8

x = 0.5*g*t**2
v = g*t

data = np.stack((x,v), axis=-1)

model = ps.SINDy(feature_names=['p', 'v'])
model.fit(data, t=t)
model.print()

# Extrapolate for further time points
t_test = np.linspace(0,4.0,400)
sim = model.simulate([0,0], t_test)

# Compute true
x_future = 0.5*g*t_test**2
v_future = g*t_test

plt.plot(t_test,x_future,label='True Position')
plt.plot(t_test,sim[:,0],"--", label='SINDy Simulated Position',linewidth=3)
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.title("True vs. SINDy Simulated Position")
plt.legend()
plt.savefig("figures/ball_pos.png")
plt.show()

plt.plot(t_test,v_future,label='True Velocity')
plt.plot(t_test,sim[:,1],"--", label='SINDy Simulated Velocity',linewidth=3)
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title("True vs. SINDy Simulated Velocity")
plt.legend()
plt.savefig("figures/ball_vel.png")
plt.show()



## What about noisy data?
t = np.linspace(0,4,400)
x = 0.5*g*t**2
v = g*t
x_noise = np.random.normal(0.0, 0.1, size=x.shape)
v_noise = np.random.normal(0.0, 0.04, size=v.shape)
noisy_x = x+x_noise
noisy_v = v+v_noise
noisy_data = np.stack((noisy_x[:200],noisy_v[:200]), axis=-1)

poly_library = ps.PolynomialLibrary(degree=2, include_interaction=False)
model_noisy = ps.SINDy(feature_library=poly_library, feature_names=['p', 'v'])
model_noisy.fit(noisy_data, t=t[:200])
model_noisy.print()

# Extrapolate for further time points
t_test = np.linspace(0,4.0,400)
sim_noisy = model_noisy.simulate([0,0], t_test)

plt.plot(t, x, label="True Measurement Position") 
plt.plot(t[:200],noisy_x[:200],label='Noisy Measurement Position')
plt.plot(t_test,sim_noisy[:,0],"--", label='SINDy Simulated Position',linewidth=3)
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.title("True vs. SINDy Simulated Position")
plt.legend()
plt.savefig("figures/NoisyBall_pos.png")
plt.show()

plt.plot(t, v, label="True Measurement Velocity") 
plt.plot(t[:200],noisy_v[:200],label='Noisy Measurement Velocity')
plt.plot(t_test, sim_noisy[:,1],"--", label='SINDy Simulated Velocity',linewidth=3)
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title("True vs. SINDy Simulated Velocity")
plt.legend()
plt.savefig("figures/NoisyBall_vel.png")
plt.show()
