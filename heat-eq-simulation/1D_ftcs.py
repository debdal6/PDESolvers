# Forward Time Centered Space (ftcs) is an explicit numerical method
# for derving the solution to PDEs

# we will use the ftcs method below to simulate the 1D Heat Equation
import numpy
import matplotlib.pyplot as pyPlot

# Global Constants

lengthOfRod = 10
maxTime = 1
heatConductivity = 1 #100 is going the least in nagtive and 1000 produces the same graph to my eye
numPointsSpace = 200
numPointsTime = 2000

# Length Vector plotted on x-axis
xSpace = numpy.linspace(0, lengthOfRod, numPointsSpace)
# Time Vector plotted on y-axis
timeSpace = numpy.linspace(0, maxTime, numPointsTime)

timeStepSize = timeSpace[1] - timeSpace[0]
spaceStepSize = xSpace[1] - xSpace[0]

boundaryConditions = [numpy.sin(lengthOfRod)+numpy.sin(maxTime), 
                    	numpy.sin(lengthOfRod)]
intialConditions = numpy.sin(xSpace)

xSpaceLength = len(xSpace)
timeSpaceLength = len(timeSpace)


# Empty Matrix/NestedList with zeroes
tempMatrix = numpy.zeros((xSpaceLength, timeSpaceLength))
tempMatrix[0,:] = boundaryConditions[0]
tempMatrix[-1,:] = boundaryConditions[1]
tempMatrix[:, 0] = intialConditions

diffusivityConstant = (heatConductivity * timeStepSize) / spaceStepSize**2
print(diffusivityConstant)

for tau in range (1, timeSpaceLength-1):
	for j in range (1, xSpaceLength-1):
		tempMatrix[j,tau] = (
				diffusivityConstant * (
					tempMatrix[j-1,tau-1] - 2 * tempMatrix[j,tau-1] + tempMatrix[j+1,tau-1]
				)
			) + tempMatrix[j,tau-1]

pyPlot.plot(tempMatrix)
pyPlot.xlabel("Length")
pyPlot.ylabel("Time")
pyPlot.title("1D Heat Diffusion Graph Plot")
pyPlot.show()