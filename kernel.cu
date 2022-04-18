#include "kernel.h"




__global__ void ScaleAndGray(unsigned char* orig, unsigned* gray, unsigned width, unsigned height, int scaleFactor) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= height || j >= width)
		return;

	int newWidth = width / scaleFactor;

	int x = (scaleFactor * i - 1 * (i > 0));
	int y = (scaleFactor * j - 1 * (j > 0));

	gray[i * newWidth + j] =
		0.3 * orig[x * (4 * width) + 4 * y] +
		0.59 * orig[x * (4 * width) + 4 * y + 1] +
		0.11 * orig[x * (4 * width) + 4 * y + 2];
}

__global__ void Zncc(unsigned* leftPixels, unsigned* rightPixels, unsigned* disparityMap,
	unsigned width, unsigned height, int minDisp, int maxDisp, int windowWidth, int windowHeight) {

	unsigned windowSize = (windowWidth+1) * (windowHeight+1) / 4;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= height || j >= width)
		return;

	float bestDisparity = maxDisp;
	float bestZncc = -1;

	// Select the best disparity value for the current pixel
	for (int d = minDisp; d <= maxDisp; d++) {
		// Calculating mean of blocks using the sliding window method
		float meanLBlock = 0, meanRBlock = 0;

		// use only other pixel value.
		for (int x = -(windowHeight-1) / 2; x <= (windowHeight-1) / 2; x+=2) {
			for (int y = -(windowWidth-1) / 2; y <= (windowWidth-1) / 2; y+=2) {
				// Check for image borders
				if (
					!(i + x >= 0) ||
					!(i + x < height) ||
					!(j + y >= 0) ||
					!(j + y < width) ||
					!(j + y - d >= 0) ||
					!(j + y - d < width)
					) {
					continue;
				}

				meanLBlock += leftPixels[(i + x) * width + (j + y)];
				meanRBlock += rightPixels[(i + x) * width + (j + y - d)];
			}
		}

		meanLBlock /= windowSize;
		meanRBlock /= windowSize;

		// Calculate ZNCC for current disparity value
		float stdLBlock = 0, stdRBlock = 0;
		float currentZncc = 0;

		for (int x = -(windowHeight - 1) / 2; x <= (windowHeight - 1) / 2; x += 2) {
			for (int y = -(windowWidth - 1) / 2; y <= (windowWidth - 1) / 2; y += 2) {
				// Check for image borders
				if (
					!(i + x >= 0) ||
					!(i + x < height) ||
					!(j + y >= 0) ||
					!(j + y < width) ||
					!(j + y - d >= 0) ||
					!(j + y - d < width)
					) {
					continue;
				}

				int centerL = leftPixels[(i + x) * width + (j + y)] - meanLBlock;
				int centerR = rightPixels[(i + x) * width + (j + y - d)] - meanRBlock;

				// standard deviation
				stdLBlock += centerL * centerL;
				stdRBlock += centerR * centerR;

				currentZncc += centerL * centerR;
			}
		}

		currentZncc /= sqrtf(stdRBlock);  // /= sqrtf(stdLBlock) * sqrtf(stdRBlock);

		// Selecting best disparity
		if (currentZncc > bestZncc) {
			bestZncc = currentZncc;
			bestDisparity = d;
		}
	}

	disparityMap[i * width + j] = (unsigned)fabs(bestDisparity);
}

__global__ void Zncc_int(unsigned* leftPixels, unsigned* rightPixels, unsigned* disparityMap,
	unsigned width, unsigned height, int minDisp, int maxDisp, int windowWidth, int windowHeight) {

	int i = blockIdx.x;  // height
	int j = threadIdx.x;  // width

	if (i >= height || j >= width)
		return;

	/*rPs_size needs to be adjusted, every time image size changes.*/
	const unsigned rPs_size = 960*6; //width*(windowHeight+1)/2
	__shared__ unsigned rPs[rPs_size];
	int sm_index = 0;
	for (int sm_i = i - (windowHeight - 1) / 2; sm_i <= i + (windowHeight - 1) / 2; sm_i += 2) {
		if ((sm_i < 0) ||
			(sm_i >= height)
			) {
			rPs[sm_index * width + j] = 0;
		}
		else {
			rPs[sm_index * width + j] = rightPixels[sm_i * width + j];
		}
		sm_index++;
	}
	__syncthreads();

	unsigned windowSize = (windowWidth + 1) * (windowHeight + 1) / 4;  // every other piexl, like chessboard.

	float bestDisparity = maxDisp;
	float bestZncc = -1;

	// Select the best disparity value for the current pixel
	for (int d = minDisp; d <= maxDisp; d++) {
		// Calculating mean of blocks using the sliding window method
		float SI = 0, SJ = 0;
		float SII = 0, SJJ = 0, SIJ = 0;

		// use only other pixel value.
		sm_index = 0;
		for (int x = -(windowHeight - 1) / 2; x <= (windowHeight - 1) / 2; x += 2) {
			for (int y = -(windowWidth - 1) / 2; y <= (windowWidth - 1) / 2; y += 2) {
				// Check for image borders
				if (
					!(i + x >= 0) ||
					!(i + x < height) ||
					!(j + y >= 0) ||
					!(j + y < width) ||
					!(j + y - d >= 0) ||
					!(j + y - d < width)
					) {
					continue;
				}

				float tempI = leftPixels[(i + x) * width + (j + y)];
				float tempJ = rPs[sm_index * width + (j + y - d)];
				//float tempJ = rightPixels[(i + x) * width + (j + y - d)];
				SI += tempI;
				SJ += tempJ;
				SII += tempI * tempI;
				SJJ += tempJ * tempJ;
				SIJ += tempI * tempJ;
			}
			sm_index++;
		}

		// (windowSize*SIJ - SI*SJ)/sqrt((windowSize*SII-SI*SI)*(windowSize * SJJ - SJ * SJ));
		float currentZncc = (windowSize*SIJ - SI*SJ)/sqrt((windowSize * SII - SI * SI) * (windowSize * SJJ - SJ * SJ));

		// Selecting best disparity
		if (currentZncc > bestZncc) {
			bestZncc = currentZncc;
			bestDisparity = d;
		}
	}

	disparityMap[i * width + j] = (unsigned)fabs(bestDisparity);
}

/* bigger disparity is better*/
__global__ void CrossCheck(unsigned* leftDisp, unsigned* rightDisp,
	unsigned* result, unsigned imSize, int crossCheckingThreshold) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= imSize)
		return;

	int diff = leftDisp[i] - rightDisp[i];
	if (diff >= 0) {  // leftDisp is winner
		if (diff <= crossCheckingThreshold) {
			result[i] = leftDisp[i];
		}
		else {
			result[i] = 0;
		}
	}
	else {  // rightDisp is winner
		if (-diff <= crossCheckingThreshold) {
			result[i] = rightDisp[i];
		}
		else {
			result[i] = 0;
		}
	}
}


__global__ void OcclusionFill(unsigned* map, unsigned* result, unsigned width, unsigned height, int occlusionNeighbours) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= height || j >= width)
		return;

	unsigned currentIndex = i * width + j;
	result[currentIndex] = map[currentIndex];

	// If the pixel value is 0, copy value from nearest non zero neighbour
	if (map[currentIndex] == 0) {
		bool stop = false;

		for (int n = 1; n <= occlusionNeighbours / 2 && !stop; n++) {
			for (int y = -n; y <= n && !stop; y++) {
				for (int x = -n; x <= n && !stop; x++) {
					// Checking for borders
					if (
						!(i + x >= 0) ||
						!(i + x < height) ||
						!(j + y >= 0) ||
						!(j + y < width) ||
						(x == 0 && y == 0)
						) {
						continue;
					}

					int index = (i + x) * width + (j + y);

					if (map[index] != 0) {
						result[currentIndex] = map[index];
						stop = true;
						break;
					}
				}
			}
		}
	}
}
