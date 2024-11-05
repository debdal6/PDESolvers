# Crank-Nicolson method is an implicit numerical method
# for derving the solution to PDEs

# we will use the crank-nicolson method below to simulate the 1D Heat Equation
import numpy
import matplotlib.pyplot as pyPlot
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Global Constants

lengthOfRod = 10
maxTime = 1
diffusivityConstant = 1
numPointsSpace = 1000
numPointsTime = 2000

# Length Vector plotted on x-axis
xDomain = numpy.linspace(0, lengthOfRod, numPointsSpace)
# Time Vector plotted on y-axis
timeDomain = numpy.linspace(0, maxTime, numPointsTime)

timeStepSize = timeDomain[1] - timeDomain[0]
spaceStepSize = xDomain[1] - xDomain[0]

# lambda functions for u0- initial condition and alpha and beta- boundary condtions
u0 = lambda x: numpy.sin(x)
alpha = lambda t: 5 * t
beta = lambda t: numpy.sin(lengthOfRod) + 2*t

#intialiizing the boundary conditions and initial condtions
boundaryConditions = numpy.array([alpha(timeDomain), beta(timeDomain)])
intialConditions = u0(xDomain)

# error assertion for intial nd boundary conditions
eps = 1e-12
err = numpy.abs(u0(0) - alpha(0))
assert(err < eps)

# Empty Matrix/NestedList with zeroes
tempMatrix = numpy.zeros((numPointsSpace, numPointsTime))
tempMatrix[0,:] = boundaryConditions[0]
tempMatrix[-1,:] = boundaryConditions[1]
tempMatrix[:, 0] = intialConditions

lambdaConstant = (diffusivityConstant * timeStepSize) / 2*(spaceStepSize**2)
print(lambdaConstant)

# Set up tridiagonal matrix coefficients
mainDiagonal = (1 + 2 * lambdaConstant) * numpy.ones(numPointsSpace - 2)
lowerDiagonal = -lambdaConstant * numpy.ones(numPointsSpace - 3)
upperDiagonal = -lambdaConstant * numpy.ones(numPointsSpace - 3)

# Create the sparse tridiagonal matrix A
A = diags([lowerDiagonal, mainDiagonal, upperDiagonal], offsets=[-1, 0, 1], format='csr')
#pyPlot.spy(A)
#pyPlot.show()

# Time-stepping loop
for tau in range(1, numPointsTime):
    # TODO: fix this, rhs equation is wrong line#64 and #65
    # Right-hand side (RHS) based on previous time step
    rhs = lambdaConstant * tempMatrix[0:-2, tau-1] + (1 - 2 * lambdaConstant) * tempMatrix[1:-1, tau-1] + lambdaConstant * tempMatrix[2:, tau-1]
    rhs[0] += lambdaConstant * tempMatrix[0, tau]    # Incorporate left boundary
    rhs[-1] += lambdaConstant * tempMatrix[-1, tau]  # Incorporate right boundary
    
    # Solve for the current time step's interior points
    solution = spsolve(A, rhs)
    # print(solution.shape)
    
    # Update the temperature matrix with the new time step values for interior points
    tempMatrix[1:-1, tau] = solution

# Visualization
# Create a meshgrid for plotting
X, Y = numpy.meshgrid(timeDomain, xDomain)

# Plot the 3D surface
fig = pyPlot.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, tempMatrix, cmap='inferno')

# Set labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Position along the rod (x)')
ax.set_zlabel('Temperature')
ax.set_title('1D Heat Diffusion Simulation')

pyPlot.show()