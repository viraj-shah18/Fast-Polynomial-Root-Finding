# Fast-Polynomial-Root-Finding

This Project was done as a part of GPU-Programming course at UC San Diego. You can have a look at [final presentation](https://docs.google.com/presentation/d/1zyHIBVsNqNs_PkI9rUWDzXf-rLHERXcp/edit?usp=drive_link&ouid=113620557867105416517&rtpof=true&sd=true) and [demo video](https://drive.google.com/file/d/15WCxTnjv5V64fRd_nJvFnwCLz2kl7uVK/view?usp=sharing)

## What is Root Finding? Why is it needed?
* Root finding basically means finding the values of a variable that satisfies an equation, making the function value zero. These solutions are known as roots or zeros of the equation.

* Whenever we are modeling any physical or real-world systems into an equation, we would often need to find the roots for the modeled equation. This is where root-finding methods are useful. 

* Easier to find roots of quadratic or at max cubic equations using direct formulas. For higher-degree functions, numerical methods are the only way to find their roots. 


## Various Root Finding Algorithms: Pros and Cons
### Bracketing method (For example Bisection Method)
+ 🟩 Guaranteed but slow convergence
- 🟥 Requires an initial interval where the function changes sign
- 🟥 Issues for roots with even multiplicity since they never change sign.
- 🟥 Returns single root.

### Non Bracketing methods (For example Secant method)
+ 🟩 Faster convergence but may not converge sometimes depending on initial guesses
- 🟥 Lot of dependence on initial guesses
- 🟥 Might miss out roots that are close to each other
- 🟥 Returns single root.


## CUDA Programming and why use it?
Several of the disadvantages on the previous slides can be addressed by running the root finding algorithm multiple times with different intervals or guess values. This way we can find all the roots for an equation as well as remove the dependency on the initial interval or guess values. Hence, clearly a use for GPU.

## Algorithm
CUDA based algorithm involves breaking a very large interval into smaller intervals and launching kernels for each smaller intervals.

For example, in bisection method, we can start with a interval of -10 to 10 and break this interval into 100 smaller intervals. Now, we run bisection method on each subinterval. This way, we can capture all the roots from -10 to 10 as well as catch roots that are close to each other like 0.1 and 0.25, which was not possible earlier even with a very small interval of 0-0.5. 

## Improvements using CUDA
### Bracketing method (For example Bisection Method)
+ 🟩 Guaranteed ~~but slow~~ convergence (Fast convergence because interval length reduces)
+ 🟩 Returns all roots. (Improved from single root)
+ 🟩 ~~Requires initial interval where the function changes sign~~ No need for specific initial interval where function changes sign
- 🟥 Issues for roots with even multiplicity since they never change sign.

### Non Bracketing methods (For example Secant method)
+ 🟩 Faster convergence
+ 🟩 Returns all roots. (Improved from single root)
+ 🟩 ~~Lot of dependence on initial guesses~~. Because of multiple kernel runs, initial guesses are not very significant now.
- 🟥 Might miss out roots that are close to each other


## Running Code
Use CMake tool to create build files. Build the C++ code. The algorithm is available as a library in Python. Have a look at py_src folder for usage instructions.

