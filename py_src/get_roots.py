import sys
sys.path.append("../build/src/pybind11_cuda_examples/Release")
import cu_root_solve as cr

import numpy as np

# Pyton inputs to C++
# can add this directly to C++ code, however, was low on time and thus completed it in python
def process_roots(output):
    roots = filter(lambda x: -10000 < x < 10000, output)
    roots = map(lambda x: round(x, 4), roots)
    roots = list(set(roots))
    return roots

low = -10
high = 10
num_intervals = 100

# bisection method
method = 0
#coeffs = np.asarray([6, -5, 1]) # 6 - 5 * x + 1 * x^2 => 2, 3
coeffs = np.asarray([-231.168, 452.3872, -322.668, 107.23, -16.8, 1]) #any complex polynomial
#coeffs = np.asarray([0.9804, -1.96, 1]) # when not on end of interval - roots get missed
#coeffs = np.asarray([0.025, -0.35, 1]) # pro - even when the roots are close enough, we can get both the roots


# secant method (not very good at finding all roots)
#method = 1
#coeffs = np.asarray([6, -5, 1]) # easy polynomial
#coeffs = np.asarray([-231.168, 452.3872, -322.668, 107.23, -16.8, 1]) #any complex polynomial

#coeffs = np.asarray([1, -2, 1]) 
#coeffs = np.asarray([0.9804, -1.98, 1]) # when not on end of interval - roots get missed
#coeffs = np.asarray([0.98, -1.98, 1])
#coeffs = np.asarray([0.025, -0.35, 1]) # pro - even when the roots are close enough, we can get both the roots


out = cr.cu_root_solve(low, high, coeffs, method, num_intervals)
roots = process_roots(out)
print(roots)