// 
// Cuda kernel for transforming a vector by a matrix of complex numbers
//
// 2022, Jonathan Tainer
//

#include "complex.h"

#ifndef MATMUL_H
#define MATMUL_H

__global__
void genmat(Complex* matrix, unsigned int N);

__global__
void matmul(Complex* matrix, float* input, Complex* output, unsigned int N);

__global__
void matmulMag(Complex* matrix, float* input, float* output, unsigned int N);

#endif
