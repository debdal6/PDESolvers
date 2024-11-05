# Forward Time Centered Space (ftcs) is an explicit numerical method
# for derving the solution to PDEs

# we will use the ftcs method below to simulate the 1D Heat Equation
import numpy
import matplotlib.pyplot as pyPlot

# Global Constants

lengthOfRod = 10
maxTime = 1
diffusivityConstant = 1
numPointsSpace = 200
numPointsTime = 2000

# Length Vector plotted on x-axis
xDomain = numpy.linspace(0, lengthOfRod, numPointsSpace)
# Time Vector plotted on y-axis
timeDomain = numpy.linspace(0, maxTime, numPointsTime)

timeStepSize = timeDomain[1] - timeDomain[0]
spaceStepSize = xDomain[1] - xDomain[0]

boundaryConditions = numpy.array([numpy.sin(lengthOfRod)+numpy.sin(maxTime), 
                    	numpy.sin(lengthOfRod)])
intialConditions = numpy.sin(xDomain)

xDomainLength = len(xDomain)
timeDomainLength = len(timeDomain)


# Empty Matrix/NestedList with zeroes
tempMatrix = numpy.zeros((xDomainLength, timeDomainLength))
tempMatrix[0,:] = boundaryConditions[0]
tempMatrix[-1,:] = boundaryConditions[1]
tempMatrix[:, 0] = intialConditions

lambdaConstant = (diffusivityConstant * timeStepSize) / spaceStepSize**2
print(lambdaConstant)

for tau in range (1, timeDomainLength-1):
	for j in range (1, xDomainLength-1):
		tempMatrix[j,tau] = (
				lambdaConstant * (
					tempMatrix[j-1,tau-1] - 2 * tempMatrix[j,tau-1] + tempMatrix[j+1,tau-1]
				)
			) + tempMatrix[j,tau-1]
print(tempMatrix)

# Create a meshgrid for plotting
X, Y = numpy.meshgrid(timeDomain, xDomain)

# Plot the 3D surface
fig = pyPlot.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, tempMatrix, cmap='inferno')

# Set labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Position along the rod')
ax.set_zlabel('Temperature')
ax.set_title('1D Heat Diffusion Simulation')

pyPlot.show()