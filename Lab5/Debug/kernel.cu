#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define NUM_BINS 4096

#define CUDA_CHECK(ans)                                                   \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
	bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
			file, line);
		if (abort)
			exit(code);
	}
}

__global__ void histo_kernel(unsigned int* input, unsigned int* bins, int inputLength){
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// All threads handle blockDim.x * gridDim.x
		// consecutive elements in each iteration
	for (unsigned int i = tid; i < inputLength; i += blockDim.x*gridDim.x){
		atomicAdd(&(bins[input[i]]), 1);
	}
	__syncthreads();
}

__global__ void histogram_privatized_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins){
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	// privatized bins
	extern __shared__ unsigned int histo_s[];

	for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x){
		histo_s[binIdx] = 0;
	}
	__syncthreads();
	
	// histogram
	for (unsigned int i = tid; i < num_elements; i += blockDim.x*gridDim.x){
		atomicAdd(&(histo_s[input[i]]), 1);
	}
	__syncthreads();

	// commit to global memory
	for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x){
		atomicAdd(&(bins[binIdx]), histo_s[binIdx]);
	}
}

__global__ void postProcess(unsigned int* bins){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < NUM_BINS){
		if (bins[tid] > 127){
			bins[tid] = 127;
		}
	}
}

int main(int argc, char *argv[]) {
	wbArg_t args;
	int inputLength;
	unsigned int *hostInput;
	unsigned int *hostBins;
	unsigned int *deviceInput;
	unsigned int *deviceBins;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
		&inputLength, "Integer");
	hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);
	wbLog(TRACE, "The number of bins is ", NUM_BINS);

	wbTime_start(GPU, "Allocating GPU memory.");
	// TODO: Allocate GPU memory here
	cudaMalloc((void **)&deviceInput, inputLength*sizeof(unsigned int));
	cudaMalloc((void **)&deviceBins, NUM_BINS*sizeof(unsigned int));
	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	// TODO: Copy memory to the GPU here
	cudaMemcpy(deviceInput, hostInput, inputLength*sizeof(unsigned int), cudaMemcpyHostToDevice);
	//cudaMemcpy(deviceBins, hostBins, NUM_BINS*sizeof(unsigned int), cudaMemcpyHostToDevice);
	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	// Launch kernel
	// ----------------------------------------------------------
	wbLog(TRACE, "Launching kernel");
	wbTime_start(Compute, "Performing CUDA computation");

	// TODO: Perform kernel computation here
	dim3 gridDim(ceil(inputLength/32)+1, 1, 1);
	dim3 blockDim(32, 1, 1);
	histo_kernel << <gridDim, blockDim >> >(deviceInput, deviceBins, inputLength);
	//histogram_privatized_kernel << <gridDim, blockDim, NUM_BINS*sizeof(unsigned int) >> >(deviceInput, deviceBins, inputLength, NUM_BINS);
	//postProcess << <gridDim, blockDim >> >(deviceBins);
	// You should call the following lines after you call the kernel.
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	// TODO: Copy the GPU memory back to the CPU here
	cudaMemcpy(hostBins, deviceBins, NUM_BINS*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	CUDA_CHECK(cudaDeviceSynchronize());
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	// TODO: Free the GPU memory here
	cudaFree(deviceBins);
	cudaFree(deviceInput);
	wbTime_stop(GPU, "Freeing GPU Memory");

	// Verify correctness
	// -----------------------------------------------------
	wbSolution(args, hostBins, NUM_BINS);

	free(hostBins);
	free(hostInput);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
