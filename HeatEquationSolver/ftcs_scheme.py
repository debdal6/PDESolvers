import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

nodes = 101     # rod is divided to 100 intervals

initial_u = np.zeros(nodes)
updated_u = np.zeros(nodes)

for i in range(1, nodes-1):
    # setting initial temp of rod to 100 (minus the first and last column)
    initial_u[i] = 100

length = 1          # rod length
dx = 1/(nodes-1)    # spatial step size
k = 0.01            # diffusivity rate
dt_max = 0.5 * (dx**2) / k   # calculating  max dt using a formula (to ensure stability)
dt = 0.2 * dt_max   # time step size

print("dx = %.2f" % dx)
print("Maximum dt = %.4f." % dt_max)
print("Used dt = %.4f." % dt)

# setting up the grid
x = np.linspace(0,1, nodes)

count = 0
count2 = 0

cmap = plt.colormaps.get_cmap('seismic_r')
colors = cmap(np.linspace(0,1,10))

# runs loop until temp in the middle of rod reaches 50
while initial_u[int(nodes/2)] > 50.0:

    for i in range(1,nodes-1):
        updated_u[i] = initial_u[i] + (dt * k * (initial_u[i-1] - 2 * initial_u[i] + initial_u[i+1]) / dx**2)

    initial_u = updated_u
    count = count + 1

    if count%1000==0:

        plt.plot(x, updated_u, color=colors[count2],
                 label="It = %g, $T_{middle}=%.2f$" % (count,initial_u[int(nodes/2)]))

        count2 = count2 + 1

plt.plot(x, initial_u, color=colors[count2],
         label="It = %g, $T_{middle}=%.2f$" % (count, initial_u[int(nodes/2)]))
plt.title("FCTS solution to heat transfer")
plt.legend(bbox_to_anchor=[1,1])
plt.xlabel("Length")
plt.ylabel("Temperature")
plt.show()
