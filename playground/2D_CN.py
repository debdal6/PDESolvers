# We will use the 5-point Laplacian aprroximation stencil explicit method 
# below to simulate the 2D Heat Equation
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Global Constants
xLength = 50
yLength = 50
maxTime = 1
diffusivityConstant = 1
numPointsSpace = 200
numPointsTime = 60

# Length Vector plotted on x-axis and y-axis
xDomain = np.linspace(0, xLength, numPointsSpace)
yDomain = np.linspace(0, yLength, numPointsSpace)
# Time Vector plotted on z-axis
timeDomain = np.linspace(0, maxTime, numPointsTime)

# Step-sizes- dt, dx, dy
timeStepSize = timeDomain[1] - timeDomain[0]
spaceStepSize = xDomain[1] - xDomain[0]

# Stability condition: dt <= dx^2/(2*k)
err_dt= np.abs((spaceStepSize**2)/(2*diffusivityConstant))
print(err_dt)
assert(timeStepSize <= err_dt)

# Calculate lambda constant for x and y (they're the same if dx = dy)
lambdaConstant = (diffusivityConstant * timeStepSize) / (spaceStepSize**2)
# Stability condition: kdt/dx^2 <= 0.5
assert(lambdaConstant <= 0.5) 
print(lambdaConstant)

# lambda functions for initial condition and boundary condtions
# 2 sin x sin 2y + 3 sin 4x sin 5y + t
u0 = lambda x, y: 2 * np.sin(x) * np.sin(2*y) + 3 * np.sin(4*x) * np.sin(5*y)
left = lambda t, y: 2 * np.sin(0) * np.sin(2*y) + 3 * np.sin(4*0) * np.sin(5*y) + t
right = lambda t, y: 2 * np.sin(xLength) * np.sin(2*y) + 3 * np.sin(4*xLength) * np.sin(5*y) + t
bottom = lambda t, x: 2 * np.sin(x) * np.sin(2*0) + 3 * np.sin(4*x) * np.sin(5*0) + t 
top = lambda t, x: 2 * np.sin(x) * np.sin(2*yLength) + 3 * np.sin(4*x) * np.sin(5*yLength) + t


# error assertion for intial & boundary conditions
err1_u0 = np.abs(u0(0, 0) - left(0, 0))
assert(err1_u0 < 1e-12)
err2_u0 = np.abs(u0(0, yLength) - right(0, 0))
assert(err2_u0 < 1e-12)
err3_u0 = np.abs(u0(0, 0) - bottom(0, 0))
assert(err3_u0 < 1e-12)
err4_u0 = np.abs(u0(0, xLength) - top(0, 0))
assert(err4_u0 < 1e-12)

def initMatrix():
    # initialize an empty matrix
    matrix = np.empty((numPointsTime, numPointsSpace, numPointsSpace))
    # set the boundary conditions for the entire 2D space
    for tau in range(numPointsTime):
        for i in range(numPointsSpace):
            matrix[tau, i, 0] = left(tau, xDomain[i])
            matrix[tau, i, -1] = right(tau, xDomain[i])
    for tau in range(numPointsTime):
        for j in range(numPointsSpace):
            matrix[tau, 0, j] = bottom(tau, yDomain[j])
            matrix[tau, -1, j] = top(tau, yDomain[j])
    # set the initial condition for the entire 2D space at t=0
    for i in range(len(xDomain)):
        for j in range(len(yDomain)):
            matrix[0,i,j] = u0(xDomain[i], yDomain[j])
    
    return matrix

def create_tridiagonal_matrix(n, lambda_const):
    """Create tridiagonal matrix for Crank-Nicolson scheme"""
    # Left hand side matrix (implicit part)
    main_diag = (1 + lambda_const) * np.ones(n)
    off_diag = (-lambda_const/2) * np.ones(n-1)
    return diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')

def solve_direction(matrix_solver, u, dt, boundary_values):
    """Solve one direction using Crank-Nicolson scheme"""
    n = u.shape[0]
    rhs = np.zeros(n)
    
    # For CN scheme, RHS includes both current time and boundary values
    lambda_const = lambdaConstant/2
    for i in range(1, n-1):
        rhs[i] = (1 - lambda_const) * u[i] + \
                 (lambda_const/2) * (u[i-1] + u[i+1])
    
    # Handle boundary conditions
    rhs[0] = boundary_values[0]
    rhs[-1] = boundary_values[1]
    
    # Solve the system
    solution = spsolve(matrix_solver, rhs)
    return solution

def calculateTemperature(U):
    # Create tridiagonal matrices for both directions using CN scheme
    n = numPointsSpace - 2  # interior points
    matrix_x = create_tridiagonal_matrix(n, lambdaConstant/2)
    matrix_y = create_tridiagonal_matrix(n, lambdaConstant/2)
    
    # Time stepping with ADI method using CN scheme
    for t in range(numPointsTime-1):
        # First half-step (x-direction)
        for j in range(1, numPointsSpace-1):
            boundary_x = [U[t+1,0,j], U[t+1,-1,j]]
            U[t+1,:,j] = solve_direction(matrix_x, U[t,:,j], timeStepSize/2, boundary_x)
        
        # Second half-step (y-direction)
        for i in range(1, numPointsSpace-1):
            boundary_y = [U[t+1,i,0], U[t+1,i,-1]]
            U[t+1,i,:] = solve_direction(matrix_y, U[t+1,i,:], timeStepSize/2, boundary_y)
    
    return U

def plot_surface(u_k, k, ax):
    ax.clear()
    X, Y = np.meshgrid(xDomain, yDomain)
    
    surf = ax.plot_surface(X, Y, u_k, 
                          cmap='jet',
                          vmin=0, vmax=100)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Temperature')
    ax.set_title(f'Temperature at t = {k*timeStepSize:.3f}')
    ax.view_init(elev=30, azim=45)
    return surf

# Create figure and 3D axes outside animation loop
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

def animate(k):
    plot_surface(tempMatrix[k], k, ax)

emptyMatrix = initMatrix()
tempMatrix = calculateTemperature(emptyMatrix)


anim = FuncAnimation(fig, animate, interval=10, frames=numPointsTime, repeat=False)
anim.save("heat_equation_solution_surf_explicit.gif", writer='pillow')