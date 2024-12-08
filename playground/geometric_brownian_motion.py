import numpy as np
import matplotlib.pyplot as plt

# defining variables
S0 = 100            # initial stock price
mu = 0.05           # drift coefficient
T = 1               # time in years
sigma = 0.03        # volatility constant (how much the stock moves)
sim = 10            # no of simulations
dt = 1/365          # daily time steps

time_steps = int(T/dt)
t = np.linspace(0, T, time_steps)

S = np.zeros((sim, time_steps))

# for all simulations at t = 0
S[:,0] = S0

for i in range(sim):
    for j in range (1, time_steps):
        # calculates based on the previous value at j-1
        S[i,j] = S[i, j-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*(np.random.normal(0, 1)))

fig = plt.figure(figsize=(10,6))
for i in range(sim):
    plt.plot(t, S[i])

plt.title("Simulated Geometric Brownian Motion")
plt.xlabel("Time (Years)")
plt.ylabel("Stock Price")
plt.show()

