#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
		    }                                                                     \
      } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//TODO: INSERT CODE HERE
__global__ void convolution_2d_tiled_kernel(float *P, const float* __restrict__ M, float *N, int height, int width, int channels) {
	// shared memory to compute output elements
	__shared__ float Ns[w][w];
	int ty = threadIdx.y;
	int	tx = threadIdx.x;
	// x and y indices for output elements
	int row_o = blockIdx.y*TILE_WIDTH + ty;
	int col_o = blockIdx.x*TILE_WIDTH + tx;
	// x and y indices for input elements
	int row_i = row_o - Mask_radius;
	int col_i = col_o - Mask_radius;

	for (int k = 0; k < channels; k++){
		float accum = 0.0f;
		// boundary checking when loading elements
		if ((row_i >= 0) && (row_i < height) && 
			(col_i >= 0) && (col_i < width)){
			Ns[ty][tx] = N[(row_i * width + col_i)* channels + k];
		}
		else{
			Ns[ty][tx] = 0.0f;
		}
		__syncthreads();

		if (ty < TILE_WIDTH && tx < TILE_WIDTH){
			for (int i = 0; i <Mask_width; i++){
				for (int j = 0; j < Mask_width; j++){
					accum += M[i*Mask_width+ j] * Ns[i + ty][j + tx];
				}
			}
			if (row_o < height && col_o < width){
				// clamp output
				P[(row_o * width + col_o)*channels + k] = clamp(accum);
			}
		}
		__syncthreads();
	}
}

int main(int argc, char *argv[]) {
	wbArg_t arg;
	int maskRows;
	int maskColumns;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	char *inputMaskFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *hostMaskData;
	float *deviceInputImageData;
	float *deviceOutputImageData;
	float *deviceMaskData;

	arg = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(arg, 0);
	inputMaskFile = wbArg_getInputFile(arg, 1);

	inputImage = wbImport(inputImageFile);
	hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

	assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
	assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	//TODO: INSERT CODE HERE
	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
	cudaMalloc((void **)&deviceMaskData, maskRows * maskColumns * sizeof(float));
	cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));

	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	//TODO: INSERT CODE HERE
	cudaMemcpy(deviceInputImageData, hostInputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData, hostMaskData,
		maskRows * maskColumns * sizeof(float),
		cudaMemcpyHostToDevice);

	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");
	//TODO: INSERT CODE HERE
	dim3 dimGrid(ceil((float)imageWidth/ TILE_WIDTH), ceil((float)imageHeight/ TILE_WIDTH),1);
	dim3 dimBlock(w, w, 1);

	convolution_2d_tiled_kernel << <dimGrid, dimBlock >> >(deviceOutputImageData, deviceMaskData, deviceInputImageData, imageHeight, imageWidth, imageChannels);

	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Doing the computation on the GPU");

	wbTime_start(Copy, "Copying data from the GPU");
	//TODO: INSERT CODE HERE
	cudaMemcpy(hostOutputImageData, deviceOutputImageData,
		imageWidth * imageHeight * imageChannels * sizeof(float),
		cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(arg, outputImage);

	//TODO: RELEASE CUDA MEMORY

	free(hostMaskData);
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
