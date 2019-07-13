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
#define clamp(x) (min(max((x), 0.0), 1.0))

__global__ void convolution(float *I, const float *M,
	float *P, int channels, int width, int height) {
	//TODO: INSERT CODE HERE

	// from pseudocode: i := col, j := row
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int row = blockIdx.y*blockDim.y + threadIdx.y;

	float accum;
	int xOffset;
	int yOffset;
	float imagePixel;
	float maskValue;
	
	if (col < width && row < height){
		int startRow = row - Mask_radius;
		int startCol = col - Mask_radius;
		for (int k = 0; k < channels; ++k){
			accum = 0;
			for (int y = 0; y < Mask_width; ++y){
				for (int x = 0; x < Mask_width; ++x){
					xOffset = startCol + x;
					yOffset = startRow + y;
					if (xOffset > -1 && xOffset < width && yOffset > -1 && yOffset < height){
						imagePixel = I[(yOffset*width + xOffset)*channels + k];
						maskValue = M[y*Mask_width + x];
						accum += imagePixel * maskValue;
					}
				}
			}
			P[(row*width + col)*channels + k] = clamp(accum);
		}
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
	cudaMalloc((void**)&deviceInputImageData, imageHeight*imageWidth*imageChannels*sizeof(float));
	cudaMalloc((void**)&deviceOutputImageData, imageHeight*imageWidth*imageChannels*sizeof(float));
	cudaMalloc((void**)&deviceMaskData, maskColumns*maskRows*sizeof(float));

	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	//TODO: INSERT CODE HERE
	cudaMemcpy(deviceInputImageData, hostInputImageData, imageChannels*imageHeight*imageWidth*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData, hostMaskData, maskColumns*maskRows*sizeof(float), cudaMemcpyHostToDevice);
	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");
	//TODO: INSERT CODE HERE
	dim3 gridDim(ceil((float)imageWidth/32), ceil((float)imageHeight/32));
	dim3 blockDim(32, 32);
	convolution << <gridDim, blockDim >> >(deviceInputImageData, deviceMaskData, deviceOutputImageData, imageChannels, imageWidth, imageHeight);
	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Doing the computation on the GPU");

	wbTime_start(Copy, "Copying data from the GPU");
	//TODO: INSERT CODE HERE
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageChannels*imageHeight*imageWidth*sizeof(float), cudaMemcpyDeviceToHost);
	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(arg, outputImage);

	//TODO: RELEASE CUDA MEMORY
	cudaFree(deviceInputImageData);
	cudaFree(deviceMaskData);
	cudaFree(deviceOutputImageData);
	free(hostMaskData);
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
