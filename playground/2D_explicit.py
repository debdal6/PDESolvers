# We will use the 5-point Laplacian aprroximation stencil explicit method 
# below to simulate the 2D Heat Equation
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as pyPlot

# Global Constants

xLength = 10
yLength = 10
maxTime = 1
diffusivityConstant = 1
numPointsSpace = 200
numPointsTime = 2000

# Length Vector plotted on x-axis
xDomain = np.linspace(0, xLength, numPointsSpace)
yDomain = np.linspace(0, yLength, numPointsSpace)
# Time Vector plotted on y-axis
timeDomain = np.linspace(0, maxTime, numPointsTime)

timeStepSize = timeDomain[1] - timeDomain[0]
spaceStepSize = xDomain[1] - xDomain[0]

# lambda functions for u0- initial condition and alpha and beta- boundary condtions
u0 = lambda n: np.sin(n)
alpha = lambda t: 5 * t
beta = lambda t: np.sin(xLength) + 2*t

# error assertion for intial nd boundary conditions
eps = 1e-12
err = np.abs(u0(0) - alpha(0))
assert(err < eps)

#intialiizing the boundary conditions and initial condtions
boundaryConditions = np.array([alpha(timeDomain), beta(timeDomain)])
intialConditions = np.array([u0(xDomain),u0(yDomain)])

xDomainLength = len(xDomain)
yDomainLength = len(yDomain)
timeDomainLength = len(timeDomain)

# Empty Matrix/NestedList with zeroes
tempMatrix = np.zeros((xDomainLength, yDomainLength, timeDomainLength))
# X-axis boundaries (first and last rows)
tempMatrix[0, :, :] = boundaryConditions[0]  # Left x boundary
tempMatrix[-1, :, :] = boundaryConditions[1]  # Right x boundary

# Y-axis boundaries (first and last columns)
tempMatrix[:, 0, :] = boundaryConditions[0]  # Bottom y boundary 
tempMatrix[:, -1, :] = boundaryConditions[1]  # Top y boundary

# Initial conditions for entire 2D space at t=0
tempMatrix[:, :, 0] = np.outer(u0(xDomain), u0(yDomain))

# Calculate lambda constants for x and y (they're the same if dx = dy)
lambdaConstant = (diffusivityConstant * timeStepSize) / spaceStepSize**2

# Stability condition for 2D heat equation (stricter than 1D)
assert(lambdaConstant < 0.25)  # Note: 0.25 instead of 0.5 for 2D
print(lambdaConstant)

# Time-stepping loop
for tau in range(1, timeDomainLength):
    # Loop over interior points (excluding boundaries)
    for i in range(1, xDomainLength-1):
        for j in range(1, yDomainLength-1):
            # 5-point stencil implementation
            tempMatrix[i,j,tau] = tempMatrix[i,j,tau-1] + lambdaConstant * (
                # x-direction terms
                (tempMatrix[i-1,j,tau-1] - 2*tempMatrix[i,j,tau-1] + tempMatrix[i+1,j,tau-1]) +
                # y-direction terms
                (tempMatrix[i,j-1,tau-1] - 2*tempMatrix[i,j,tau-1] + tempMatrix[i,j+1,tau-1])
            )

print(tempMatrix)

# # Create meshgrid for x and y coordinates
# X, Y = np.meshgrid(xDomain, yDomain)

# # Create the figure and 3D axes
# fig = pyPlot.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')

# # Function to update the plot for animation
# def update_plot(frame):
#     ax.clear()
    
#     # Plot the surface for the current time step
#     surf = ax.plot_surface(X, Y, tempMatrix[:,:,frame], 
#                           cmap='inferno',
#                           vmin=np.min(tempMatrix),
#                           vmax=np.max(tempMatrix))
    
#     # Set labels and title
#     ax.set_xlabel('X Position')
#     ax.set_ylabel('Y Position')
#     ax.set_zlabel('Temperature')
#     ax.set_title(f'2D Heat Diffusion at t = {timeDomain[frame]:.3f}')
    
#     return surf,

# # Create animation
# anim = animation.FuncAnimation(fig, update_plot, 
#                              frames=range(0, timeDomainLength, 20),  # Animate every 20th frame
#                              interval=50,  # 50ms between frames
#                              blit=True)

# # Add a colorbar
# pyPlot.colorbar(ax.plot_surface(X, Y, tempMatrix[:,:,0], cmap='inferno'))

# pyPlot.show()