# by Monika Lopatniuk and Luca Marie Mahlke

import numpy as np
import matplotlib.pyplot as plt

# if we would have actual data e.g. in .csv format we would use: 
''' import pandas as pd
    data = pd.read_csv("aquired_data")
    x = data("time").to_numpy()
    y = data("signal").to_numpy()
    
    dx = x(1) - x(0)                # to set distance between to data points
    n_samples = len(x)
    L = n_samples * dx (...)'''

# set parameters
n_samples = 500                     # number of data points the discrete signal has
dx = 2 * np.pi / n_samples          # for spacing between each point
L = n_samples * dx                  # domain length
N = 10                              # sets number of loops of the aproximation

k = np.arange (1, n_samples + 1)    # creates one‑dimensional NumPy array with evenly spaced numbers from 1 to n 
x = k * dx                          # sets last equation on page 3

np.random.seed(0)                   # fixes the random numbers to get the same result for every run         
f_data = np.sin(x) + 0.5 * np.cos( 3 * x) + 0.4 * np.random.randn(n_samples)    # equation 7 and np.random.randn(n_samples) is added to simulate real-world discrete data

# implementing a0, A_n and B_n using np.sum for equation 9, 10 and 11
def a0_discrete(f):
    return (1 / L) * np.sum(f) * dx

def A_n_discrete(f, n):
    omega_n = 2 * np.pi * n / L
    return (2 / L) * np.sum (f * np.sin(omega_n * x)) * dx

def B_n_discrete(f, n):
    omega_n = 2 * np.pi * n / L
    return (2 / L) * np.sum (f * np.cos(omega_n * x)) * dx


# as in part 1
a0_val = a0_discrete(f_data)
A = np.array ([A_n_discrete(f_data, n) for n in range (1, N+1)]) # builds array for the sine coefficient
B = np.array ([B_n_discrete(f_data, n) for n in range (1, N+1)]) # builds array for the cosine coefficient

def reconstructed(xx):
    s = a0_val * np.ones_like(xx)       # np.ones_like(xx) creates an array of ones with the same shape as xx > multiplied by a0_val that gives the constant baseline
    for n in range (1, N + 1):
        omega_n = 2 * np.pi * n / L     # as part 1
        s += A[n-1] * np.sin( omega_n * xx ) + B[n-1] * np.cos( omega_n * xx)  # equation 1
    return s

x_plot = np.linspace(dx, L, 1000)
f_reconstructed = reconstructed(x_plot)

plt.figure(figsize=(10, 5)) # to set width and height of the plot
plt.plot(x, f_data, label = "Original noisy data", color = "grey")                               # plots f(x) > the original function, for all 1000 x_plot values
plt.plot( x_plot, f_reconstructed, "--", label = f"Fourier reconstruction (N={N})", color = "red")  # plots the fourrier approximation as a dashed line
plt.xlabel("x")     # labels the x axis
plt.ylabel("f(x)")  # labels the y axis
plt.legend()        # plots a leged using the label terms from above
plt.grid(True)      # plots grid
plt.title("Part 2: Fourier Coefficients from Discrete Time-Series Data", weight = "bold")             # sets title of the plot
plt.show()          # needed to actually show the plot