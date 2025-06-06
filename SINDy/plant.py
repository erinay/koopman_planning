import numpy as np
g = 9.8
l = 1.0

def pendulum(t,z):
    theta, omega = z
    return [omega, -g/l*np.sin(theta)]