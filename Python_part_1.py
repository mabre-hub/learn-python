import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# set parameters
L = 2 * np.pi       # upper limit of the interval
N = 10              # test what changes with this number? sets number of loops of the aproximation

# functions 
# def f(x):         # equation 6
#    return x        

def f(x):           # equation 7
    return np.sin(x) + 0.5 * np.cos(3*x)        # could also define it as a middle step and return e.g. f(x), but this is easier to read

# circle requence
def omega(n):       # equation 2
    return 2 * np.pi * n / L

# mean value 
def a0():           # equation 3
    value,_ = integrate.quad (lambda x: f(x), 0, L)       # _ to ignore the error value, quad normally returns integral value and estimated error
    return value / L                            

# sine coefficient
def A_n(n):           # equation 4
    value,_ = integrate.quad (lambda x: f(x) * np.sin( omega(n) * x ), 0, L)
    return 2 / L * value

# cosine coefficient
def B_n(n):           # equation 5
    value,_ = integrate.quad (lambda x: f(x) * np.cos( omega(n) * x ), 0, L)
    return 2 / L * value

# all basics for the formula representing the Fourier Series 
a_val = a0()        # sets contant coefficient (average value of f(x) over a period)
A = np.array ([A_n(n) for n in range (1, N+1)]) # builds array for the sine coefficient
B = np.array ([B_n(n) for n in range (1, N+1)]) # builds array for the cosine coefficient

def fourier_approx(x):
    result = a_val # if we's use a0() here not only the definition but also a0() would get calculated multiple times but is only needed once
    for n in range (1, N+1):                    # loops from 1 no N
        result += A[n-1] * np.sin( omega(n) * x ) + B[n-1] * np.cos( omega(n) * x)  # equation 1
    return result

x_plot = np.linspace ( 0, L, 1000)  # creates 1000 evenly spaced values between 0 and L

plt.plot( x_plot, f(x_plot), label = "Original f(x)")                               # plots f(x) > the original function, for all 1000 x_plot values
plt.plot( x_plot, fourier_approx(x_plot), "--", label = f"Fourier Approx (N={N})")  # plots the fourrier approximation as a dashed line
plt.xlabel("x")     # labels the x axis
plt.ylabel("f(x)")  # labels the y axis
plt.legend()        # plots a leged using the label terms from above
plt.title("Fourier-Series approximation (Part 1: Analytical Function)")             # sets title of the plot
plt.show()          # needed to actually show the plot