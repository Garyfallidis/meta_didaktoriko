from libc.math cimport sin

cpdef double sinc_kernel(double x):
    if -0.01 < x < 0.01:
        return 1.0
    return sin(x) / x

def sinc_kernel(x):

	cdef: 
		double x_tmp = x

	if -0.01 < x_tmp < 0.01:
        return 1.0
    return sin(x_tmp) / x_tmp		