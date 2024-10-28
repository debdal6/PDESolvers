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
xs = numpy.linspace(0, lengthOfRod, numPointsSpace)
# Time Vector plotted on y-axis
ts = numpy.linspace(0, maxTime, numPointsTime)

timeStepSize = ts[1] - ts[0]
spaceStepSize = xs[1] - xs[0]

boundaryConditions = [numpy.sin(lengthOfRod)+numpy.sin(maxTime), 
                    numpy.sin(lengthOfRod)]
intialConditions = numpy.sin(xs)

xsLength = len(xs)
tsLength = len(ts)


# Empty Matrix/NestedList with zeroes
gridMatrix = numpy.zeros((xsLength, tsLength))
gridMatrix[0,:] = boundaryConditions[0]
gridMatrix[-1,:] = boundaryConditions[1]
gridMatrix[:, 0] = intialConditions

factor=(heatConductivity*timeStepSize)/spaceStepSize**2
print(factor)

for tau in range (1, tsLength-1):
    for j in range (1, xsLength-1):
        
        gridMatrix[j,tau]=(factor*
                (gridMatrix[j-1,tau-1]-
                 2*gridMatrix[j,tau-1]+
                 gridMatrix[j+1,tau-1])
            ) + gridMatrix[j,tau-1]


pyPlot.plot(gridMatrix)
pyPlot.show()