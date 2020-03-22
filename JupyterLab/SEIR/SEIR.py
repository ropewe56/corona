import numpy as np
import pylab as plt


def SEIR_step(i, S, E, I, R, N, a, b, g, d, dt):
    dd = b * E[i] * S[i] / N
    dS = - dd
    dE =   dd - a * E[i-d]
    dI =        a * E[i-d] - g * I[i]
    dR =                     g * I[i]
    
    S[i+1] = S[i] + dS * dt
    E[i+1] = E[i] + dE * dt
    I[i+1] = I[i] + dI * dt
    R[i+1] = R[i] + dR * dt

n_days = 365
n_steps_per_day = 10

n = n_days * n_steps_per_day

dt = 1.0/float(n_steps_per_day)

N = 6.0e6#8.0e7
S = np.zeros(n) + N
E = np.ones(n)
I = np.zeros(n)
R = np.zeros(n)

a = 0.143
b = 0.25
g = 0.11
d = 7
for i in range(d, (n)-1):
    SEIR_step(i, S, E, I, R, N, a, b, g, d, dt)

t = np.mgrid[0.0:float(n_days):(n)*1j]

plt.plot(t, S, 'g', label="S")
plt.plot(t, E, 'b', label="E")
plt.plot(t, I, 'r', label="I")
plt.plot(t, R, 'm', label="R")
plt.figure()
plt.plot(t, E, 'b', label="E")
plt.plot(t, I, 'r', label="I")

plt.legend()
plt.show()