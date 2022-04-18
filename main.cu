#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"
#include "lodepng.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <cassert>







/*
Class to calculate time taken by functions in seconds.
* Creating an object of the class in a function, calls the constructor which starts the timer.
* At the end of the function, the destructor is called which stops the timer and calculates the duration.
* We can get the duration manually using the getElapsedTime method.
*/
class Timer {
private:
	std::chrono::time_point<std::chrono::steady_clock> m_Start, m_End;
	std::chrono::duration<float> m_Duration;

public:
	Timer() {
		m_Start = std::chrono::high_resolution_clock::now();
	}

	~Timer() {
		m_End = std::chrono::high_resolution_clock::now();
		m_Duration = m_End - m_Start;

		std::cout << "Done (" << m_Duration.count() << " s)" << std::endl;
	}

	float getElapsedTime() {
		m_End = std::chrono::high_resolution_clock::now();
		m_Duration = m_End - m_Start;

		return m_Duration.count();
	}
};


// Display GPU info
// https://stackoverflow.com/a/5689133
void DisplayHeader() {
	const int kb = 1024;
	const int mb = kb * kb;
	std::cout << "NBody.GPU" << std::endl << "=========" << std::endl << std::endl;

	std::cout << "CUDA version:   v" << CUDART_VERSION << std::endl;

	int devCount;
	cudaGetDeviceCount(&devCount);
	std::cout << "CUDA Devices: " << std::endl << std::endl;

	for (int i = 0; i < devCount; ++i) {
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		std::cout << i << ": " << props.name << ": " << props.major << "." << props.minor << std::endl;
		std::cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << std::endl;
		std::cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << std::endl;
		std::cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << std::endl;
		std::cout << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;

		std::cout << "  Warp size:         " << props.warpSize << std::endl;
		std::cout << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
		std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
		std::cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << std::endl;
		std::cout << std::endl;
	}
}

std::vector<unsigned char> loadImage(const char* filename, unsigned& width, unsigned& height) {
	Timer timer;

	std::vector<unsigned char> pixels;

	unsigned error = lodepng::decode(pixels, width, height, filename);
	if (error) {
		std::cout << "Failed to load image: " << lodepng_error_text(error) << std::endl;
		std::cin.get();
		exit(-1);
	}

	return pixels;
}

std::vector<unsigned char> normalize(std::vector<unsigned> in, const unsigned width, const unsigned height) {
	std::vector<unsigned char> result(width * height * 4);

	unsigned char max = 0;
	unsigned char min = UCHAR_MAX;

	for (int i = 0; i < width * height; i++) {
		if (in[i] > max) {
			max = in[i];
		}

		if (in[i] < min) {
			min = in[i];
		}
	}

	// Normalize values to be between 0 and 255
	int mapIndex = 0;
	for (int i = 0; i < width * height * 4; i += 4, mapIndex++) {
		result[i] = result[i + 1] = result[i + 2] = (unsigned char)(255 * (in[mapIndex] - min) / (max - min));
		result[i + 3] = 255;
	}

	return result;
}


void CudaCall(const cudaError_t& status) {
	if (status != cudaSuccess) {
		std::cout << "Error [" << status << "]: " << cudaGetErrorString(status) << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl;
	}
}

constexpr int scaleFactor = 2;

constexpr int minDisparity = 0;
constexpr int maxDisparity = 64;
constexpr int windowWidth = 11;
constexpr int windowHeight = 11;

constexpr int crossCheckingThreshold = 2;

constexpr int occlusionNeighbours = 256;

int main() {
	Timer timer;

	DisplayHeader();

	// Host variables
	std::vector<unsigned char> leftPixels, rightPixels;  // 1 byte: 0-255
	unsigned width, height, rightWidth, rightHeight;

	std::cout << "Reading Left Image...";
	leftPixels = loadImage("realL.png", width, height);

	std::cout << "Reading Right Image...";
	rightPixels = loadImage("realR.png", rightWidth, rightHeight);

	// left and right images are assumed to be of same dimensions
	assert(width == rightWidth && height == rightHeight);

	width /= scaleFactor;
	height /= scaleFactor;

	unsigned imSize = width * height;
	unsigned origSize = rightWidth * rightHeight;
	std::vector<unsigned> output(imSize);

	// Device variabels
	unsigned char* d_origL, * d_origR;
	unsigned* d_grayL, * d_grayR, * d_dispLR, * d_dispRL, * d_dispCC, * d_output;

	CudaCall(cudaMalloc((void**)&d_origL, sizeof(unsigned char) * origSize * 4));  // 4: rgbd.
	CudaCall(cudaMalloc((void**)&d_origR, sizeof(unsigned char) * origSize * 4));
	CudaCall(cudaMalloc((void**)&d_grayL, sizeof(unsigned) * imSize));
	CudaCall(cudaMalloc((void**)&d_grayR, sizeof(unsigned) * imSize));
	CudaCall(cudaMalloc((void**)&d_dispLR, sizeof(unsigned) * imSize));
	CudaCall(cudaMalloc((void**)&d_dispRL, sizeof(unsigned) * imSize));
	CudaCall(cudaMalloc((void**)&d_dispCC, sizeof(unsigned) * imSize));
	CudaCall(cudaMalloc((void**)&d_output, sizeof(unsigned) * imSize));

	// Copy Data from host to device
	CudaCall(cudaMemcpy(d_origL, leftPixels.data(), sizeof(leftPixels[0]) * leftPixels.size(), cudaMemcpyHostToDevice));
	CudaCall(cudaMemcpy(d_origR, rightPixels.data(), sizeof(rightPixels[0]) * rightPixels.size(), cudaMemcpyHostToDevice));

	// Profiling
	float elapsed = 0;
	cudaEvent_t start, stop;

	CudaCall(cudaEventCreate(&start));
	CudaCall(cudaEventCreate(&stop));

	// Kernel Calls
	dim3 blocks(height / 21, width / 21);
	dim3 threads(21, 21);
	dim3 blocks1D((height / 21) * (width / 21));
	dim3 threads1D(21 * 21);

	// Scale and Gray left
	std::cout << "Converting Left Image to grayscale...";
	CudaCall(cudaEventRecord(start));

	ScaleAndGray<<<blocks, threads>>>(d_origL, d_grayL, rightWidth, rightHeight, scaleFactor);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// Scale and Gray right
	std::cout << "Converting Right Image to grayscale...";
	CudaCall(cudaEventRecord(start));

	ScaleAndGray<<<blocks, threads>>>(d_origR, d_grayR, rightWidth, rightHeight, scaleFactor);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// Disparity Left over Right
	std::cout << "Converting Left Disparity Map...";
	CudaCall(cudaEventRecord(start));

	Zncc<<<blocks, threads>>>(d_grayL, d_grayR, d_dispLR, width, height, minDisparity, maxDisparity, windowWidth, windowHeight);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// Disparity Right over Left
	std::cout << "Converting Right Disparity Map...";
	CudaCall(cudaEventRecord(start));

	Zncc<<<blocks, threads>>>(d_grayR, d_grayL, d_dispRL, width, height, -maxDisparity, -minDisparity, windowWidth, windowHeight);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// Cross Checking
	std::cout << "Performing Cross Checking...";
	CudaCall(cudaEventRecord(start));

	CrossCheck<<<blocks1D, threads1D>>>(d_dispLR, d_dispRL, d_dispCC, imSize, crossCheckingThreshold);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// Occlusion Filling
	std::cout << "Performing Occlusion Filling...";
	CudaCall(cudaEventRecord(start));

	OcclusionFill<<<blocks, threads>>>(d_dispCC, d_output, width, height, occlusionNeighbours);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// Copy data from device to host
	CudaCall(cudaMemcpy(&output[0], d_output, sizeof(unsigned)* imSize, cudaMemcpyDeviceToHost));

	lodepng::encode("output.png", normalize(output, width, height), width, height);

	std::cout << "The program took " << timer.getElapsedTime() << " s" << std::endl;

	cudaFree(d_origL);
	cudaFree(d_origR);
	cudaFree(d_grayL);
	cudaFree(d_grayR);
	cudaFree(d_dispLR);
	cudaFree(d_dispRL);
	cudaFree(d_dispCC);
	cudaFree(d_output);

	std::cin.get();
	return 0;
}


