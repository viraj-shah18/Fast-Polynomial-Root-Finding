/*************************************************************************
/* Author: Viraj Shah
/* Email : vishah@ucsd.edu
/* University of California, San Diego
/*************************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>

#define NUM_STREAMS 4
#define THREADS_PER_BLOCK 16
#define DEFAULT_NO_ROOT 10000
#define maxIterations 100

__global__ void getRootBisection(double* low, double* high, double* out, double* poly, int degree);
__device__ double getFunctionValue(double* poly, int degree, double x);


__device__ double getFunctionValue(double* poly, int degree, double x) {
	/**
	 * @brief Finds the value of a polynomial at a given point.
	 *
	 * This function calculates the value of a polynomial at a given point x.
	 * 
	 * @param poly The polynomial to be evaluated. The polynomial is stored as an array of doubles, where the index of the array is the power of the variable.
	 * @param degree The degree of the polynomial.
	 * @param x The point at which the polynomial is evaluated.
	 * @return The value of the polynomial at the given point.
	*/
	

	// calculate value of polynomial at x
	double value = 0;
	for (int i = degree; i >= 0; i--) {
		value = value * x + poly[i];
	}

	return value;
}


__global__ void getRootBisection(double* low, double* high, double* out, double* poly, int degree) {
	// the function finds the root of a polynomial using the bisection method
	/**
	* @brief Calculates the root of a function using the bisection method.
	*
	* @param low The lower bound of the interval in which the root is to be found.
	* @param high The upper bound of the interval in which the root is to be found.
	* @param out The output array in which the root is to be stored.
	* @param poly The polynomial whose root is to be found. The polynomial is stored as an array of doubles, where the index of the array is the power of the variable.
	* @param degree The degree of the polynomial.
	* @return None
	*/
	
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	double thread_low = low[idx];
	double thread_high = high[idx];

	double fn_low = getFunctionValue(poly, degree, thread_low);
	double fn_high = getFunctionValue(poly, degree, thread_high);

	// check for change of signs 
	//	- if no change -> add -1 and return
	if (fn_low * fn_high > 0) {
		out[idx] = DEFAULT_NO_ROOT;
		return;
	}

	if (fn_low == 0) {
		out[idx] = thread_low;
		return;
	}
	if (fn_high == 0) {
		out[idx] = thread_high;
		return;
	}

	// run while loop till convergence
	double mid = 0;
	double tol = 1e-5;
	while (abs(thread_high-thread_low) > tol) {
		mid = (thread_low + thread_high) / 2;

		if (getFunctionValue(poly, degree, thread_low) * getFunctionValue(poly, degree, mid) < 0) {
			thread_high = mid;
		}
		else {
			thread_low = mid;
		}
	}

	// add value to output
	out[idx] = mid;

	return;
}


__global__ void getRootSecant(double* low, double* high, double* h_out, double* poly, int degree) {
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	double o_low = low[idx];
	double o_high = high[idx];

	
	double curr = low[idx];
	double prev = high[idx];
	double fn_curr, fn_prev;

	double tol = 1e-5;
	int iteration = 0;
	while (iteration < maxIterations) {
		fn_curr = getFunctionValue(poly, degree, curr); 
		fn_prev = getFunctionValue(poly, degree, prev);

		if (curr < o_low || curr > o_high) {
			h_out[idx] = DEFAULT_NO_ROOT;
			return;
		}
		
		if (abs(curr - prev) < tol || abs(fn_curr) < tol || iteration >= maxIterations) {
			break;
		}
		
		double next_val = curr - fn_curr * (curr - prev) / (fn_curr - fn_prev);

		prev = curr;
		curr = next_val;

		iteration++;
	}

	h_out[idx] = curr;

}



void cu_root_solve(double low, double high, double* poly, double* h_out, int method, int degree, int num_intervals) {
	/**
	* @brief Finds the roots of a polynomial.
	*
	* This function finds the roots of a polynomial using the bisection or secant method.
	*
	* @param low The lower bound of the interval in which the root is to be found.
	* @param high The upper bound of the interval in which the root is to be found.
	* @param poly The polynomial whose root is to be found. The polynomial is stored as an array of doubles, where the index of the array is the power of the variable.
	* @param h_out array to store the output roots
	* @param degree The degree of the polynomial.
	* @param method The method to be used to find the root. The method can be either BISECTION or SECANT.
	* @return None
	**/

	double* d_low;
	double* d_high;
	double* d_out;
	double* d_poly;

	double* h_low;
	double* h_high;

	int size = sizeof(double) * num_intervals;
	int poly_size = sizeof(double) * (degree + 1);


	cudaHostAlloc((void**)&h_low, size, cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_high, size, cudaHostAllocDefault);

	cudaMalloc((void**)&d_low, size);
	cudaMalloc((void**)&d_high, size);
	cudaMalloc((void**)&d_out, size);
	cudaMalloc((void**)&d_poly, poly_size);

	// creating intervals in host
	double interval_length = (high - low) / num_intervals;
	for (int i = 0; i < num_intervals; i++) {
		h_low[i] = low + i * interval_length;
		h_high[i] = low + (i + 1) * interval_length;
	}

	// create 4 cuda streams
	cudaStream_t streams[NUM_STREAMS];
	int nsdata = num_intervals / NUM_STREAMS;
	size_t iBytes = nsdata * sizeof(double);
	size_t polyBytes = (degree + 1) * sizeof(double);

	dim3 block(THREADS_PER_BLOCK);
	dim3 grid((nsdata + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

	for (int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamCreate(&streams[i]);
	}

	for (int i = 0; i < NUM_STREAMS; ++i) {
		int offset = i * nsdata;
		cudaMemcpyAsync(&d_low[offset], &h_low[offset], iBytes, cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(&d_high[offset], &h_high[offset], iBytes, cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(d_poly, poly, polyBytes, cudaMemcpyHostToDevice, streams[i]);

		// run kernel code -> depending on method specified
		if (method == 0) {
			getRootBisection << <grid, block, 0, streams[i] >> > (&d_low[offset], &d_high[offset], &d_out[offset], d_poly, degree);
		}
		else if (method == 1) {
			getRootSecant <<<grid, block, 0, streams[i] >>> (&d_low[offset], &d_high[offset], &d_out[offset], d_poly, degree);
		}
		else {
			printf("Invalid method specified. Exiting.\n");
			exit(1);
		}

		cudaMemcpyAsync(&h_out[offset], &d_out[offset], iBytes, cudaMemcpyDeviceToHost, streams[i]);

	}

	for (int i = 0; i < NUM_STREAMS; ++i) {
		cudaStreamSynchronize(streams[i]);
	}

	for (int i = 0; i < NUM_STREAMS; ++i) {
		cudaStreamDestroy(streams[i]);
	}

	cudaFreeHost(h_low);
	cudaFreeHost(h_high);

	cudaFree(d_low);
	cudaFree(d_high);
	cudaFree(d_out);
	cudaFree(d_poly);
}
