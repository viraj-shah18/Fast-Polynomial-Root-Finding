/*************************************************************************
/* Author: Viraj Shah
/* Email : vishah@ucsd.edu
/* University of California, San Diego
/*************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

extern void cu_root_solve(double low, double high, double* coefficients, double* roots, int method, int degree, int num_intervals);

namespace py = pybind11;

py::array_t<double> root_finder_wrapper(double low, double high, py::array_t<double> coeff, int method, int num_intervals ){
	auto buf = coeff.request();
	
	if (buf.ndim != 1)
		throw std::runtime_error("Number of dimensions must be one");
	
	int N = coeff.shape()[0];
	int degree = N-1;
	
	auto result = py::array(py::buffer_info(nullptr, sizeof(double), py::format_descriptor<double>::value, buf.ndim, { num_intervals }, { sizeof(double) }));
	
	auto buf2 = result.request();
	double* coefficients = (double*)buf.ptr;
	double* roots = (double*) buf2.ptr;
	cu_root_solve(low, high, coefficients, roots, method, degree, num_intervals);
	return result;
}

PYBIND11_MODULE(cu_root_solve, m) {
    m.def("cu_root_solve", &root_finder_wrapper, "Find all the roots for a polynomial functions. ");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
