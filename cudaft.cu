// 
// DFT class implementation
// 
// 2022, Jonathan Tainer
// 

#include "cudaft.h"
#include "complex.h"
#include "matmul.h"

#include <cuda.h>
#include <stdlib.h>
#include <math.h>

CudaFT::CudaFT() {
	N = 0;
	devMatrix = NULL;
	devInput = NULL;
	devOutput = NULL;
}

CudaFT::~CudaFT() {
	cudaFree(devMatrix);
	cudaFree(devInput);
	cudaFree(devOutput);
}

void CudaFT::setDims(unsigned int n) {
	
	// Allocate or reallocate an appropriate amount of GPU memory
	cudaFree(devMatrix);
	cudaFree(devInput);
	cudaFree(devOutput);

	N = n;

	cudaMalloc((void**)&devMatrix, sizeof(Complex) * N * N);
	cudaMalloc((void**)&devInput, sizeof(float) * N);
	cudaMalloc((void**)&devOutput, sizeof(Complex) * N);

	// Call kernel to construct DFT matrix in GPU memory
	genmat<<<(N * N / 256) + 1, 256>>>(devMatrix, N);

}

void CudaFT::transform(float* inputBuffer, Complex* outputBuffer) {
	
	// Copy input buffer to GPU memory
	cudaMemcpy(devInput, inputBuffer, sizeof(float) * N, cudaMemcpyHostToDevice);
	
	// Spawn kernel threads
	matmul<<<(N / 256) + 1, 256>>>(devMatrix, devInput, devOutput, N);
	
	// Copy output buffer from GPU memory to system memory
	cudaMemcpy(outputBuffer, devOutput, sizeof(Complex) * N, cudaMemcpyDeviceToHost);
}

void CudaFT::transformMag(float* inputBuffer, float* outputBuffer) {
	// Copy input buffer to GPU memory
	cudaMemcpy(devInput, inputBuffer, sizeof(float) * N, cudaMemcpyHostToDevice);

	// Spawn kernel threads
	matmulMag<<<(N / 256) + 1, 256>>>(devMatrix, devInput, (float*) devOutput, N);

	// Copy output buffer from GPU memory to system memory
	cudaMemcpy(outputBuffer, devOutput, sizeof(float) * N, cudaMemcpyDeviceToHost);
}















