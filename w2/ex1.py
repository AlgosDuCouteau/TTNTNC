import polars as pl
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.00001
df = pl.read_csv('data.csv')
x = df.select('x').to_numpy().reshape(-1, 2)
x_mat = df.select(pl.lit(1).alias('ones'), 'x').to_numpy().reshape(-1, 2)
y = df.select('y').to_numpy().reshape(-1, 1)
m = len(y)
theta = np.random.rand(2, 1)
H = np.matmul(x_mat, theta)

loop = 10
J_values = np.zeros((loop, 1))
for i in range(loop):
    theta = theta - (alpha/m)*np.matmul(x_mat.T, (H-y))
    H = np.matmul(x_mat, theta)
    J_values[i] = (1/(2*m))*np.matmul((H-y).T, (H-y))
    
# Create a figure and a set of subplots
fig, axs = plt.subplots(2)

# Plot the cost function on the first subplot
axs[0].plot(J_values)
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Cost (J)')
axs[0].set_title('Cost Function')

# Plot the line and (x, y) on the second subplot
axs[1].scatter(x, y)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = theta[1]*x_line + theta[0]
axs[1].plot(y_line, 'r-')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].set_title('Function')

# Layout so plots do not overlap
fig.tight_layout()

plt.show()