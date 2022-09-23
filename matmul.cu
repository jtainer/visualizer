// 
// Cuda kernel for transforming a vector by a matrix of complex numbers
//
// 2022, Jonathan Tainer
//

#include "matmul.h"
#include "complex.h"

__global__
void genmat(Complex* matrix, unsigned int N) {
	
	// Determine thread ID
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	
	// Ensure excess threads do nothing
	if (tid < N * N) {
	
		// Find XY coords of the location of the current thread in the matrix
		int x = tid % N;
		int y = tid / N;
		
		// Calculate omega for given N
		Complex omega;
		omega.real = cosf(-2 * M_PI / N);
		omega.imag = sinf(-2 * M_PI / N);
		
		Complex prod = { 1, 0 };

		for (unsigned int i = 0; i < x * y; i++) {
			Complex tmp;

			tmp.real = (prod.real * omega.real) - (prod.imag * omega.imag);
			tmp.imag = (prod.real * omega.imag) + (prod.imag * omega.real);

			prod = tmp;
		}

		prod.real /= sqrtf((float) N);
		prod.imag /= sqrtf((float) N);

		matrix[tid] = prod;	
	}
}

__global__
void matmul(Complex* matrix, float* input, Complex* output, unsigned int N) {
	
	// Determine the thread ID
	int y = threadIdx.x + (blockIdx.x * blockDim.x);

	// Ensure excess threads do nothing
	if (y < N) {
		
		// Multiply input vector by current row in DFT matrix
		Complex sum = { 0, 0 };

		for (unsigned int x = 0; x < N; x++) {
			sum.real += matrix[(y * N) + x].real * input[x];
			sum.imag += matrix[(y * N) + x].imag * input[x];
		}

		output[y] = sum;
	}
}

__global__
void matmulMag(Complex* matrix, float* input, float* output, unsigned int N) {

	// Determine the thread ID
	int y = threadIdx.x + (blockIdx.x * blockDim.x);

	// Ensure excess threads do nothing
	if (y < N) {
		
		// Multiply input vector by current row in DFT matrix
		Complex sum = { 0, 0 };

		for (unsigned int x = 0; x < N; x++) {
			sum.real += matrix[(y * N) + x].real * input[x];
			sum.imag += matrix[(y * N) + x].imag * input[x];
		}

		// Compute magnitude of the phasor
		output[y] = sqrt((sum.real * sum.real) + (sum.imag * sum.imag));
	}
}
